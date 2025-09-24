#!/usr/bin/env python3
"""
FastAPI web application for the AI-powered product price checker.
"""

import asyncio
import json
import logging
import math
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from queue import Queue
from threading import Thread
import time

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Spacer
from reportlab.lib.units import inch

from .config import OUTPUT_FILE, DEFAULT_WORKER_POOL_SIZE, MAX_WORKER_POOL_SIZE
from .smart_rate_limiter import global_rate_manager
from .spreadsheet_io import SpreadsheetHandler
from .matcher import ProductMatcher
from .colored_logging import setup_colored_logging

# Setup logging
setup_colored_logging(verbose=True)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AI Product Price Checker", version="1.0.0")

# Global storage for session data and queues
sessions: Dict[str, Dict] = {}
processing_queues: Dict[str, Queue] = {}  # session_id -> Queue of items to process
queue_status: Dict[str, Dict] = {}  # session_id -> {current_item, total_items, processed_items}
processing_threads: Dict[str, Thread] = {}  # session_id -> processing thread
queued_items: Dict[str, set] = {}  # session_id -> set of item_numbers that are queued
processing_items: Dict[str, set] = {}  # session_id -> set of item_numbers currently being processed

# Create directories if they don't exist
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
uploads_dir = Path("uploads")
screenshots_dir = Path("screenshots")

for directory in [templates_dir, static_dir, uploads_dir, screenshots_dir]:
    directory.mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/screenshots", StaticFiles(directory=str(screenshots_dir)), name="screenshots")
templates = Jinja2Templates(directory=str(templates_dir))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and parse an Excel file."""
    if not file.filename.endswith(('.xlsx', '.xls', '.xlsm')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed")
    
    try:
        # Create a session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        upload_path = uploads_dir / f"{session_id}_{file.filename}"
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse the spreadsheet
        spreadsheet = SpreadsheetHandler(str(upload_path))
        df = spreadsheet.load_spreadsheet()
        items = spreadsheet.get_product_items()
        
        # Store session data
        sessions[session_id] = {
            "file_path": str(upload_path),
            "spreadsheet": spreadsheet,
            "items": items,
            "df": df,
            "url_mapping": {},
            "llm_price_mapping": {},
            "screenshot_data": {},
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Initialize queue status
        queue_status[session_id] = {
            "current_item": None,
            "total_items": len(items),
            "processed_items": 0,
            "is_processing": False,
            "completed": False
        }
        
        # Initialize queued items tracking
        queued_items[session_id] = set()
        
        # Convert DataFrame to dict for JSON response
        items_data = []
        for item in items:
            # Clean numeric values to handle NaN/infinity
            def clean_numeric(value, default=0):
                if value is None or pd.isna(value):
                    return default
                if isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        return default
                    return float(value) if isinstance(value, float) else int(value)
                return default
            
            # Clean string values to handle None/NaN
            def clean_string(value, default=""):
                if value is None or pd.isna(value):
                    return default
                return str(value)
            
            item_dict = {
                "item_number": clean_numeric(item.get("item_number", 0), 0),
                "room": clean_string(item.get("room", "")),
                "make": clean_string(item.get("make", "")),
                "model": clean_string(item.get("model", "")),
                "description": clean_string(item.get("description", "")),
                "original_vendor": clean_string(item.get("original_vendor", "")),
                "quantity_lost": clean_numeric(item.get("quantity_lost", 1), 1),
                "item_age_months": clean_numeric(item.get("item_age_months", None), None),
                "condition": clean_string(item.get("condition", "")),
                "cost_to_replace_each": clean_numeric(item.get("cost_to_replace_each", 0), 0),
                "total_cost": clean_numeric(item.get("total_cost", 0), 0),
                "price": clean_numeric(item.get("price", 0), 0),
                "matched_url": "",
                "llm_extracted_price": "",
                "fallback_extracted_price": "",
                "extracted_title": "",
                "screenshot_path": "",
                "is_processing": False
            }
            items_data.append(item_dict)
        
        return {
            "session_id": session_id,
            "items": items_data,
            "filename": file.filename,
            "total_items": len(items)
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def process_queue_worker(session_id: str):
    """Worker function that processes items from the queue one by one."""
    try:
        logger.info(f"🔄 Starting worker for session {session_id[:8]}")
        session = sessions[session_id]
        queue = processing_queues[session_id]
        status = queue_status[session_id]
        
        # Initialize matcher
        matcher = ProductMatcher()
        
        while True:
            try:
                # Get next item from queue (blocks until available)
                item = queue.get(timeout=1.0)  # 1 second timeout
                
                if item is None:  # Sentinel value to stop processing
                    logger.info(f"⏹️ Worker stopping for session {session_id[:8]}")
                    break
                
                item_number = item.get("item_number")
                logger.info(f"🛍️ Processing item {item_number}")
                
                # Update status and remove from queued items
                status["current_item"] = item_number
                status["is_processing"] = True
                if session_id in queued_items and item_number in queued_items[session_id]:
                    queued_items[session_id].remove(item_number)
                
                # Process single item using the matcher
                try:
                    # Use the single item processing method
                    url_mapping, llm_price_mapping, screenshot_data = matcher.process_single_item(item)
                    
                    # Update session data with results
                    session["url_mapping"].update(url_mapping)
                    session["llm_price_mapping"].update(llm_price_mapping)
                    session["screenshot_data"].update(screenshot_data)
                    
                    logger.info(f"✅ Completed item {item_number}")
                    
                except Exception as e:
                    logger.error(f"Error processing item {item_number}: {e}")
                    # Add error results
                    session["url_mapping"][item_number] = "ERROR"
                    session["llm_price_mapping"][item_number] = ""
                    session["screenshot_data"][item_number] = {
                        "screenshot_path": "",
                        "price": "",
                        "title": "",
                        "error": str(e)
                    }
                
                # Update progress
                status["processed_items"] += 1
                status["current_item"] = None
                status["is_processing"] = False
                
                # Mark task as done
                queue.task_done()
                
                logger.info(f"Progress: {status['processed_items']}/{status['total_items']} items completed")
                
            except Exception as e:
                if "timeout" not in str(e).lower():
                    logger.error(f"Error in queue worker: {e}")
                # Check if we should continue processing
                if not queue.empty():
                    continue
                else:
                    # No more items and we got a timeout, likely done
                    break
        
        # Mark processing as complete
        status["completed"] = True
        status["is_processing"] = False
        status["current_item"] = None
        
        logger.info(f"🎉 Session {session_id[:8]} complete: {status['processed_items']} items processed")
        
    except Exception as e:
        logger.error(f"Fatal error in queue worker for session {session_id}: {e}")
        # Mark as completed even on error to prevent hanging
        if session_id in queue_status:
            queue_status[session_id]["completed"] = True
            queue_status[session_id]["is_processing"] = False
            queue_status[session_id]["current_item"] = None

@app.post("/process_all/{session_id}")
async def process_all_items(session_id: str):
    """Process all items using domain-queued approach."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    items = session["items"]
    
    # Reset status
    queue_status[session_id] = {
        "current_item": None,
        "total_items": len(items),
        "processed_items": 0,
        "is_processing": True,
        "completed": False
    }
    
    # Start async processing with configurable concurrency
    async def process_all_async():
        try:
            # Create a semaphore to limit concurrent processing (configurable)
            worker_pool_size = min(DEFAULT_WORKER_POOL_SIZE, len(items))  # Don't create more workers than items
            
            logger.info(f"Starting limited-concurrency processing for session {session_id} with {len(items)} items (max {worker_pool_size} concurrent)")
            
            # Track progress by running LLM calls with individual completion tracking
            completed_items = 0
            
            # Initialize tracking sets
            if session_id not in queued_items:
                queued_items[session_id] = set()
            if session_id not in processing_items:
                processing_items[session_id] = set()
            
            semaphore = asyncio.Semaphore(worker_pool_size)
            logger.info(f"Using worker pool size: {worker_pool_size}")
            
            # Mark all items as queued initially
            for item in items:
                item_number = item.get("item_number")
                queued_items[session_id].add(item_number)
            
            async def process_item_with_progress(item):
                nonlocal completed_items
                item_number = item.get("item_number")
                
                # Acquire semaphore to limit concurrency
                async with semaphore:
                    try:
                        # Mark item as currently processing (remove from queued, add to processing)
                        if item_number in queued_items[session_id]:
                            queued_items[session_id].remove(item_number)
                        processing_items[session_id].add(item_number)
                        
                        logger.info(f"🔄 Processing item {item_number} (slot acquired, {len(processing_items[session_id])} active)")
                        
                        # Create individual matcher for each item to avoid race conditions
                        matcher = ProductMatcher()
                        
                        try:
                            # Process single item  
                            result = await matcher._find_url_for_item(item)
                            item_number, url, price, screenshot_data = result
                            
                            # Update session data immediately as each item completes
                            session["url_mapping"][item_number] = url or "NOT_FOUND"
                            session["llm_price_mapping"][item_number] = price or ""
                            session["screenshot_data"][item_number] = screenshot_data or {}
                            
                            # Mark item as no longer processing
                            processing_items[session_id].discard(item_number)
                        finally:
                            # Clean up this item's matcher resources
                            await matcher.search_agent.cleanup_screenshotter()
                        
                        # Update progress
                        completed_items += 1
                        queue_status[session_id]["processed_items"] = completed_items
                        
                        logger.info(f"✓ Completed item {item_number} ({completed_items}/{len(items)}) - slot released")
                        return result
                        
                    except Exception as e:
                        logger.error(f"Error processing item {item.get('item_number')}: {e}")
                        session["url_mapping"][item_number] = "ERROR"
                        session["llm_price_mapping"][item_number] = ""
                        session["screenshot_data"][item_number] = {"error": str(e)}
                        
                        # Mark item as no longer processing
                        processing_items[session_id].discard(item_number)
                        completed_items += 1
                        queue_status[session_id]["processed_items"] = completed_items
                        return (item_number, None, None, None)
            
            # Process items with limited concurrency (max {worker_pool_size} at a time)
            logger.info(f"Starting processing with max {worker_pool_size} concurrent tasks...")
            tasks = [process_item_with_progress(item) for item in items]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Clear any remaining queued items and update final status
            if session_id in queued_items:
                queued_items[session_id].clear()
            
            queue_status[session_id].update({
                "processed_items": len(items),
                "is_processing": False,
                "completed": True,
                "current_item": None
            })
            
            # No need to cleanup - each item had its own matcher that cleaned up individually
            
            logger.info(f"✓ Completed processing all {len(items)} items for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error in process_all_async for session {session_id}: {e}")
            # Mark as completed with error
            queue_status[session_id].update({
                "is_processing": False,
                "completed": True,
                "current_item": None
            })
    
    # Run async processing in background thread  
    def run_async_in_thread():
        import asyncio
        try:
            # Try to get existing event loop, create new one if needed
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create task in it
                asyncio.create_task(process_all_async())
            else:
                loop.run_until_complete(process_all_async())
        except RuntimeError:
            # No event loop, create new one
            asyncio.run(process_all_async())
    
    thread = Thread(target=run_async_in_thread, daemon=True)
    processing_threads[session_id] = thread
    thread.start()
    
    logger.info(f"Started limited-concurrency processing for session {session_id} with {len(items)} items (max {DEFAULT_WORKER_POOL_SIZE} concurrent)")
    
    return {"message": f"Limited-concurrency processing started (max {DEFAULT_WORKER_POOL_SIZE} concurrent)", "session_id": session_id, "total_items": len(items)}

@app.post("/process_item/{session_id}/{item_number}")
async def process_single_item(session_id: str, item_number: int):
    """Process a single item using domain-queued approach."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Find the specific item
    target_item = None
    for item in session["items"]:
        if item.get("item_number") == item_number:
            target_item = item
            break
    
    if not target_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Initialize tracking sets for this session
    if session_id not in processing_items:
        processing_items[session_id] = set()
    if session_id not in queued_items:
        queued_items[session_id] = set()
    
    # Mark this specific item as processing
    processing_items[session_id].add(item_number)
    
    # Process single item with domain queuing and status updates
    async def process_single_async():
        try:
            logger.info(f"Starting domain-queued processing for item {item_number} in session {session_id}")
            
            # Use the new domain-queued matcher
            matcher = ProductMatcher()
            
            # Process single item with status tracking
            result = await matcher._find_url_for_item(target_item)
            item_num, url, price, screenshot_data = result
            
            # Update session data with results
            session["url_mapping"][item_number] = url or "NOT_FOUND"
            session["llm_price_mapping"][item_number] = price or ""
            session["screenshot_data"][item_number] = screenshot_data or {}
            
            # Remove item from processing set
            processing_items[session_id].discard(item_number)
            
            # Clean up LLM resources
            await matcher.search_agent.cleanup_screenshotter()
            
            logger.info(f"✓ Completed processing item {item_number} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error processing item {item_number} for session {session_id}: {e}")
            # Add error results
            session["url_mapping"][item_number] = "ERROR"
            session["llm_price_mapping"][item_number] = ""
            session["screenshot_data"][item_number] = {
                "screenshot_path": "",
                "price": "",
                "title": "",
                "error": str(e)
            }
            # Remove item from processing set even on error
            processing_items[session_id].discard(item_number)
    
    # Run async processing in background thread
    def run_async_in_thread():
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(process_single_async())
            else:
                loop.run_until_complete(process_single_async())
        except RuntimeError:
            asyncio.run(process_single_async())
    
    thread = Thread(target=run_async_in_thread, daemon=True)
    processing_threads[session_id] = thread
    thread.start()
    
    logger.info(f"Started domain-queued processing for item {item_number} in session {session_id}")
    
    return {"message": f"Item {item_number} processing started with domain queuing", "session_id": session_id}

@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of a session including queue progress."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    status = queue_status.get(session_id, {
        "current_item": None,
        "total_items": len(session["items"]),
        "processed_items": 0,
        "is_processing": False,
        "completed": True
    })
    
    # Build updated items list
    items_data = []
    for item in session["items"]:
        item_number = item.get("item_number")
        
        # Clean numeric values to handle NaN/infinity
        def clean_numeric(value, default=0):
            if value is None or pd.isna(value):
                return default
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    return default
                return float(value) if isinstance(value, float) else int(value)
            return default
        
        # Clean string values to handle None/NaN
        def clean_string(value, default=""):
            if value is None or pd.isna(value):
                return default
            return str(value)
        
        # Check item status: queued, processing, or completed
        is_queued = session_id in queued_items and item_number in queued_items[session_id]
        has_been_processed = item_number in session["url_mapping"]
        is_currently_processing = session_id in processing_items and item_number in processing_items[session_id]
        
        # Item is "processing" if it's queued OR currently being processed
        is_processing = is_queued or is_currently_processing
        
        item_dict = {
            "item_number": clean_numeric(item_number, 0),
            "room": clean_string(item.get("room", "")),
            "make": clean_string(item.get("make", "")),
            "model": clean_string(item.get("model", "")),
            "description": clean_string(item.get("description", "")),
            "original_vendor": clean_string(item.get("original_vendor", "")),
            "quantity_lost": clean_numeric(item.get("quantity_lost", 1), 1),
            "item_age_months": clean_numeric(item.get("item_age_months", None), None),
            "condition": clean_string(item.get("condition", "")),
            "cost_to_replace_each": clean_numeric(item.get("cost_to_replace_each", 0), 0),
            "total_cost": clean_numeric(item.get("total_cost", 0), 0),
            "price": clean_numeric(item.get("price", 0), 0),
            "matched_url": clean_string(session["url_mapping"].get(item_number, "")),
            "llm_extracted_price": clean_string(session["llm_price_mapping"].get(item_number, "")),
            "fallback_extracted_price": clean_string(session["screenshot_data"].get(item_number, {}).get("price", "")),
            "extracted_title": clean_string(session["screenshot_data"].get(item_number, {}).get("title", "")),
            "screenshot_path": clean_string(session["screenshot_data"].get(item_number, {}).get("screenshot_path", "")),
            "is_processing": is_processing,
            "is_queued": is_queued,
            "has_been_processed": has_been_processed
        }
        items_data.append(item_dict)
    
    return {
        "session_id": session_id,
        "items": items_data,
        "total_items": len(session["items"]),
        "processed_urls": len([url for url in session["url_mapping"].values() if url and url not in ["NOT_FOUND", "ERROR"]]),
        "processed_screenshots": len([data for data in session["screenshot_data"].values() if data.get("screenshot_path")]),
        "queue_status": {
            "current_item": status["current_item"],
            "total_items": status["total_items"],
            "processed_items": status["processed_items"],
            "is_processing": status["is_processing"],
            "completed": status["completed"],
            "queue_size": 0,  # No longer using queues - domain queuing happens internally
            "progress_percentage": round((status["processed_items"] / max(status["total_items"], 1)) * 100, 1) if status["total_items"] > 0 else 0
        }
    }

@app.get("/rate_limiter_metrics")
async def get_rate_limiter_metrics():
    """Get smart rate limiter metrics for all providers."""
    try:
        metrics = global_rate_manager.get_all_metrics()
        return {
            "status": "success",
            "timestamp": time.time(),
            "metrics": metrics,
            "summary": {
                "total_providers": len(metrics),
                "active_circuits": sum(1 for m in metrics.values() if m["circuit_state"] == "closed"),
                "open_circuits": sum(1 for m in metrics.values() if m["circuit_state"] == "open"),
                "average_success_rate": round(
                    sum(m["success_rate"] for m in metrics.values()) / len(metrics), 2
                ) if metrics else 0,
                "total_requests": sum(m["total_requests"] for m in metrics.values())
            }
        }
    except Exception as e:
        logger.error(f"Error getting rate limiter metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/download_pdf/{session_id}")
async def download_images_pdf(session_id: str):
    """Generate and download a PDF with all screenshots."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    screenshot_data = session["screenshot_data"]
    
    if not screenshot_data:
        raise HTTPException(status_code=400, detail="No screenshots available")
    
    try:
        # Create PDF
        pdf_path = uploads_dir / f"{session_id}_screenshots.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story = []
        
        for item_number, data in screenshot_data.items():
            screenshot_path = data.get("screenshot_path")
            if screenshot_path and Path(screenshot_path).exists():
                # Add item info
                from reportlab.platypus import Paragraph
                from reportlab.lib.styles import getSampleStyleSheet
                styles = getSampleStyleSheet()
                
                story.append(Paragraph(f"Item {item_number}: {data.get('extracted_title', 'No title')}", styles['Title']))
                story.append(Spacer(1, 12))
                
                # Add image
                img = Image(screenshot_path, width=7*inch, height=5*inch)
                story.append(img)
                story.append(Spacer(1, 24))
        
        if story:
            doc.build(story)
            return FileResponse(
                path=str(pdf_path),
                filename=f"screenshots_{session_id}.pdf",
                media_type="application/pdf"
            )
        else:
            raise HTTPException(status_code=400, detail="No valid screenshots found")
            
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@app.get("/debug/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to check session state."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    status = queue_status.get(session_id, {})
    queue = processing_queues.get(session_id)
    
    return {
        "session_id": session_id,
        "queue_status": status,
        "queue_size": queue.qsize() if queue else 0,
        "queued_items": list(queued_items.get(session_id, set())),
        "processed_urls": len(session["url_mapping"]),
        "processed_screenshots": len(session["screenshot_data"]),
        "current_processing": status.get("current_item"),
        "is_processing": status.get("is_processing", False),
        "completed": status.get("completed", False)
    }

@app.get("/download_excel/{session_id}")
async def download_excel(session_id: str):
    """Download the updated Excel file."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    try:
        # Update spreadsheet with results
        spreadsheet = session["spreadsheet"]
        spreadsheet.add_url_column(session["url_mapping"])
        spreadsheet.add_llm_prices(session["llm_price_mapping"])
        spreadsheet.add_screenshot_data(session["screenshot_data"])
        
        # Save to temporary file
        output_path = uploads_dir / f"{session_id}_results.xlsx"
        spreadsheet.save_spreadsheet(str(output_path))
        
        return FileResponse(
            path=str(output_path),
            filename=f"results_{session_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        logger.error(f"Error generating Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")

@app.post("/stop_processing/{session_id}")
async def stop_processing(session_id: str):
    """Stop processing for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_id in processing_queues:
        # Add sentinel value to stop the worker
        processing_queues[session_id].put(None)
        logger.info(f"Stop signal sent to queue worker for session {session_id}")
    
    if session_id in queue_status:
        queue_status[session_id]["completed"] = True
        queue_status[session_id]["is_processing"] = False
        queue_status[session_id]["current_item"] = None
    
    # Clear queued items
    if session_id in queued_items:
        queued_items[session_id].clear()
    
    return {"message": "Processing stopped", "session_id": session_id}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up all associated data."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Stop processing if running
    if session_id in processing_queues:
        processing_queues[session_id].put(None)  # Send stop signal
        del processing_queues[session_id]
    
    # Clean up status tracking
    if session_id in queue_status:
        del queue_status[session_id]
    
    # Clean up queued items tracking
    if session_id in queued_items:
        del queued_items[session_id]
    
    # Clean up thread reference (thread will die naturally when function returns)
    if session_id in processing_threads:
        del processing_threads[session_id]
    
    # Clean up uploaded file
    session = sessions[session_id]
    file_path = Path(session["file_path"])
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Deleted uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file {file_path}: {e}")
    
    # Delete session
    del sessions[session_id]
    
    logger.info(f"Session {session_id} deleted and cleaned up")
    
    return {"message": f"Session {session_id} deleted", "session_id": session_id}

@app.get("/healthz")
async def healthcheck():
    """Simple health check endpoint for container monitoring."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 