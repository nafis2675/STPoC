// Object Detection App JavaScript - Two Image Workflow
class ObjectDetectionApp {
    constructor() {
        this.currentStep = 1; // 1 = first image, 2 = second image, 3 = comparison
        this.currentStream = null;
        this.workflowState = {
            image1: null,
            image2: null,
            comparison: null
        };
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateParameterValues();
        this.setupPasteHandler();
        this.updateWorkflowUI();
    }

    bindEvents() {
        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Camera button
        document.getElementById('camera-btn').addEventListener('click', () => {
            this.openCamera();
        });

        // Paste button
        document.getElementById('paste-btn').addEventListener('click', () => {
            this.handlePaste();
        });

        // Camera modal controls
        document.getElementById('capture-btn').addEventListener('click', () => {
            this.capturePhoto();
        });

        document.getElementById('camera-close-btn').addEventListener('click', () => {
            this.closeCamera();
        });

        // Modal close
        document.querySelector('.close').addEventListener('click', () => {
            this.closeCamera();
        });

        // Parameter sliders
        document.getElementById('conf-threshold').addEventListener('input', (e) => {
            document.getElementById('conf-value').textContent = e.target.value;
        });

        document.getElementById('iou-threshold').addEventListener('input', (e) => {
            document.getElementById('iou-value').textContent = e.target.value;
        });

        // New parameter adjustment sliders
        document.getElementById('new-conf-threshold').addEventListener('input', (e) => {
            document.getElementById('new-conf-value').textContent = e.target.value;
        });

        document.getElementById('new-iou-threshold').addEventListener('input', (e) => {
            document.getElementById('new-iou-value').textContent = e.target.value;
        });

        // Workflow navigation buttons
        document.getElementById('continue-btn').addEventListener('click', () => {
            this.continueToNextStep();
        });

        document.getElementById('view-comparison-btn').addEventListener('click', () => {
            this.showComparison();
        });

        // Parameter adjustment buttons
        document.getElementById('adjust-parameters-btn').addEventListener('click', () => {
            this.adjustParameters();
        });

        document.getElementById('skip-adjustment-btn').addEventListener('click', () => {
            this.showResultsSaving();
        });

        document.getElementById('adjust-params-from-comparison-btn').addEventListener('click', () => {
            this.showParameterAdjustment();
        });

        // Results saving buttons
        document.getElementById('download-report-btn').addEventListener('click', () => {
            this.downloadReport();
        });

        document.getElementById('download-images-btn').addEventListener('click', () => {
            this.downloadImages();
        });

        document.getElementById('start-over-btn').addEventListener('click', () => {
            this.resetWorkflow();
        });

        document.getElementById('save-from-comparison-btn').addEventListener('click', () => {
            this.showResultsSaving();
        });

        document.getElementById('restart-workflow-btn').addEventListener('click', () => {
            this.resetWorkflow();
        });

        // Click outside modal to close
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('camera-modal');
            if (e.target === modal) {
                this.closeCamera();
            }
        });
    }

    updateParameterValues() {
        const confSlider = document.getElementById('conf-threshold');
        const iouSlider = document.getElementById('iou-threshold');
        const newConfSlider = document.getElementById('new-conf-threshold');
        const newIouSlider = document.getElementById('new-iou-threshold');
        
        document.getElementById('conf-value').textContent = confSlider.value;
        document.getElementById('iou-value').textContent = iouSlider.value;
        document.getElementById('new-conf-value').textContent = newConfSlider.value;
        document.getElementById('new-iou-value').textContent = newIouSlider.value;
    }

    setupPasteHandler() {
        // Global paste event listener
        document.addEventListener('paste', (e) => {
            const items = e.clipboardData.items;
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const blob = items[i].getAsFile();
                    this.handleFileUpload(blob);
                    break;
                }
            }
        });
    }

    updateWorkflowUI() {
        // Update step indicators
        document.querySelectorAll('.step').forEach(step => step.classList.remove('active', 'completed'));
        
        if (this.currentStep >= 1) {
            const step1 = document.getElementById('step1');
            if (this.workflowState.image1) {
                step1.classList.add('completed');
            } else {
                step1.classList.add('active');
            }
        }

        if (this.currentStep >= 2) {
            const step2 = document.getElementById('step2');
            if (this.workflowState.image2) {
                step2.classList.add('completed');
            } else {
                step2.classList.add('active');
            }
        }

        if (this.currentStep >= 3) {
            document.getElementById('step3').classList.add('active');
        }

        // Update step info
        const stepInfo = document.getElementById('step-info');
        const uploadText = document.getElementById('upload-text');
        const cameraText = document.getElementById('camera-text');
        const pasteText = document.getElementById('paste-text');

        if (this.currentStep === 1) {
            stepInfo.innerHTML = `
                <h2><i class="fas fa-arrow-right"></i> Step 1: Upload First Image</h2>
                <p>Start by uploading, capturing, or pasting your first image for object detection.</p>
            `;
            uploadText.textContent = 'Upload First Image';
            cameraText.textContent = 'Take First Photo';
            pasteText.textContent = 'Paste First Image';
        } else if (this.currentStep === 2) {
            stepInfo.innerHTML = `
                <h2><i class="fas fa-arrow-right"></i> Step 2: Upload Second Image</h2>
                <p>Now upload, capture, or paste your second image to compare with the first.</p>
            `;
            uploadText.textContent = 'Upload Second Image';
            cameraText.textContent = 'Take Second Photo';
            pasteText.textContent = 'Paste Second Image';
        } else {
            stepInfo.innerHTML = `
                <h2><i class="fas fa-check-circle"></i> Both Images Processed</h2>
                <p>Both images have been analyzed. View the comparison results below.</p>
            `;
        }
    }

    async handleFileUpload(file) {
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this.showToast('Please select a valid image file', 'error');
            return;
        }

        // Convert file to base64 for processing
        const base64 = await this.fileToBase64(file);
        await this.processImage(base64, file.name);
    }

    async handlePaste() {
        try {
            const permission = await navigator.permissions.query({ name: 'clipboard-read' });
            if (permission.state === 'granted' || permission.state === 'prompt') {
                const clipboardItems = await navigator.clipboard.read();
                for (const clipboardItem of clipboardItems) {
                    for (const type of clipboardItem.types) {
                        if (type.startsWith('image/')) {
                            const blob = await clipboardItem.getType(type);
                            this.handleFileUpload(blob);
                            return;
                        }
                    }
                }
            }
            this.showToast('No image found in clipboard. Copy an image and try again.', 'warning');
        } catch (err) {
            this.showToast('Clipboard access failed. Use Ctrl+V to paste images directly.', 'warning');
        }
    }

    async openCamera() {
        // Check if we're on HTTPS or localhost
        const isSecureContext = window.isSecureContext;
        const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        
        // Check if we're on mobile
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        if (isMobile && !isSecureContext && !isLocalhost) {
            this.showToast(
                'Camera access requires HTTPS on mobile devices. Please access via HTTPS or use the file upload option instead.', 
                'error'
            );
            this.showMobileHTTPSHelp();
            return;
        }
        
        try {
            // Check if getUserMedia is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia not supported');
            }
            
            // Mobile-friendly camera constraints
            const constraints = isMobile ? {
                video: {
                    facingMode: 'environment', // Use back camera on mobile
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 }
                }
            } : {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            this.currentStream = stream;
            const video = document.getElementById('camera-video');
            video.srcObject = stream;
            
            document.getElementById('camera-modal').style.display = 'block';
            this.showToast('Camera opened successfully', 'success');
        } catch (err) {
            console.error('Camera error:', err);
            
            let errorMessage = 'Could not access camera.';
            
            if (err.name === 'NotAllowedError') {
                errorMessage = 'Camera access denied. Please allow camera permission and try again.';
            } else if (err.name === 'NotFoundError') {
                errorMessage = 'No camera found on this device.';
            } else if (err.name === 'NotSupportedError') {
                if (isMobile && !isSecureContext) {
                    errorMessage = 'Camera requires HTTPS on mobile. Please use file upload instead.';
                    this.showMobileHTTPSHelp();
                } else {
                    errorMessage = 'Camera not supported by this browser.';
                }
            } else if (err.name === 'NotReadableError') {
                errorMessage = 'Camera is being used by another application.';
            } else {
                if (isMobile && !isSecureContext) {
                    errorMessage = 'Camera access requires HTTPS on mobile devices. Please use file upload instead.';
                    this.showMobileHTTPSHelp();
                }
            }
            
            this.showToast(errorMessage, 'error');
        }
    }

    closeCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }
        document.getElementById('camera-modal').style.display = 'none';
    }

    capturePhoto() {
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            await this.handleFileUpload(blob);
            this.closeCamera();
        }, 'image/jpeg', 0.9);
    }

    async processImage(imageData, fileName = 'captured_image') {
        this.showLoading(true);
        
        try {
            const params = this.getDetectionParameters();
            const imageNumber = this.currentStep;
            
            const response = await fetch(`/detect-base64/${imageNumber}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    ...params
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                // Store result in workflow state
                this.workflowState[`image${imageNumber}`] = result;
                
                this.displayCurrentImageResult(result, imageNumber);
                this.showToast(`Image ${imageNumber} processed successfully!`, 'success');
                
                // Check if ready for comparison
                if (result.ready_for_comparison) {
                    this.currentStep = 3;
                    this.updateWorkflowUI();
                }
            } else {
                throw new Error(result.message || 'Detection failed');
            }
        } catch (error) {
            console.error('Detection error:', error);
            this.showToast(`Detection failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    getDetectionParameters() {
        return {
            conf_threshold: parseFloat(document.getElementById('conf-threshold').value),
            iou_threshold: parseFloat(document.getElementById('iou-threshold').value),
            max_detection: parseInt(document.getElementById('max-detection').value)
        };
    }

    displayCurrentImageResult(result, imageNumber) {
        const { report, original_image, annotated_image } = result;

        // Show individual results section
        document.getElementById('individual-results').style.display = 'block';
        document.getElementById('both-images-summary').style.display = 'none';
        document.getElementById('comparison-section').style.display = 'none';

        // Update title and images
        document.getElementById('current-image-title').textContent = `Image ${imageNumber}`;
        document.getElementById('current-original-image').src = original_image;
        document.getElementById('current-annotated-image').src = annotated_image;

        // Display summary stats
        document.getElementById('current-total-objects').textContent = report.total_objects;
        document.getElementById('current-unique-classes').textContent = report.unique_classes;

        // Display class counts
        this.displayClassCounts(report.class_counts, 'current-class-counts-list');

        // Display detailed detections
        this.displayDetections(report.detections, 'current-detections-table');

        // Update continue button
        const continueBtn = document.getElementById('continue-btn');
        const continueText = document.getElementById('continue-text');
        
        if (imageNumber === 1) {
            continueText.textContent = 'Continue to Image 2';
            continueBtn.style.display = 'inline-flex';
        } else {
            continueText.textContent = 'View Summary';
            continueBtn.style.display = 'inline-flex';
        }

        // Scroll to results
        document.getElementById('individual-results').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    continueToNextStep() {
        if (this.currentStep === 1) {
            this.currentStep = 2;
            this.updateWorkflowUI();
            // Hide results and reset file input
            document.getElementById('individual-results').style.display = 'none';
            document.getElementById('file-input').value = '';
        } else {
            this.showBothImagesSummary();
        }
    }

    showBothImagesSummary() {
        // Show both images summary
        document.getElementById('individual-results').style.display = 'none';
        document.getElementById('both-images-summary').style.display = 'block';

        const image1Result = this.workflowState.image1;
        const image2Result = this.workflowState.image2;

        // Display summary images
        document.getElementById('summary-image1-annotated').src = image1Result.annotated_image;
        document.getElementById('summary-image2-annotated').src = image2Result.annotated_image;

        // Display object counts
        document.getElementById('summary-image1-objects').textContent = 
            `${image1Result.report.total_objects} objects`;
        document.getElementById('summary-image2-objects').textContent = 
            `${image2Result.report.total_objects} objects`;

        // Scroll to summary
        document.getElementById('both-images-summary').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    async showComparison() {
        this.showLoading(true);
        
        try {
            const response = await fetch('/compare', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.workflowState.comparison = result.comparison;
                this.displayComparison(result.comparison, result.image1, result.image2);
                this.showToast('Comparison completed successfully!', 'success');
            } else {
                throw new Error(result.message || 'Comparison failed');
            }
        } catch (error) {
            console.error('Comparison error:', error);
            this.showToast(`Comparison failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayComparison(comparison, result1, result2) {
        // Show comparison section
        document.getElementById('both-images-summary').style.display = 'none';
        document.getElementById('comparison-section').style.display = 'block';

        // Display comparison images
        document.getElementById('compare-image1').src = result1.annotated_image;
        document.getElementById('compare-image2').src = result2.annotated_image;
        
        document.getElementById('compare-objects1').textContent = `${result1.report.total_objects} objects`;
        document.getElementById('compare-objects2').textContent = `${result2.report.total_objects} objects`;

        // Display object change
        const change = comparison.total_objects.change;
        const changeElement = document.getElementById('object-change');
        changeElement.textContent = change >= 0 ? `+${change}` : `${change}`;

        // Display class changes
        this.displayClassChanges(comparison.class_changes);

        // Display new and removed classes
        this.displayNewClasses(comparison.new_classes);
        this.displayRemovedClasses(comparison.removed_classes);

        // Scroll to comparison
        document.getElementById('comparison-section').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    showParameterAdjustment() {
        document.getElementById('comparison-section').style.display = 'none';
        document.getElementById('parameter-adjustment').style.display = 'block';
        
        document.getElementById('parameter-adjustment').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    async adjustParameters() {
        this.showLoading(true);
        
        try {
            const newParams = {
                conf_threshold: parseFloat(document.getElementById('new-conf-threshold').value),
                iou_threshold: parseFloat(document.getElementById('new-iou-threshold').value),
                max_detection: parseInt(document.getElementById('new-max-detection').value)
            };

            const response = await fetch('/adjust-parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newParams)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                // Update workflow state with adjusted results
                this.workflowState.image1 = result.adjusted_results.image1;
                this.workflowState.image2 = result.adjusted_results.image2;
                this.workflowState.comparison = result.comparison;

                // Show updated comparison
                this.displayComparison(result.comparison, 
                    result.adjusted_results.image1, 
                    result.adjusted_results.image2);
                
                this.showToast('Parameters adjusted successfully!', 'success');
            } else {
                throw new Error(result.message || 'Parameter adjustment failed');
            }
        } catch (error) {
            console.error('Parameter adjustment error:', error);
            this.showToast(`Parameter adjustment failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async showResultsSaving() {
        document.getElementById('comparison-section').style.display = 'none';
        document.getElementById('parameter-adjustment').style.display = 'none';
        document.getElementById('results-saving').style.display = 'block';

        try {
            const response = await fetch('/save-results', {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                if (result.success) {
                    document.getElementById('report-preview').textContent = result.report_text;
                    this.savedResults = result;
                }
            }
        } catch (error) {
            console.error('Error preparing results:', error);
        }

        document.getElementById('results-saving').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    downloadReport() {
        if (this.savedResults) {
            const blob = new Blob([this.savedResults.report_text], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'object_detection_report.txt';
            a.click();
            URL.revokeObjectURL(url);
            this.showToast('Report downloaded successfully!', 'success');
        }
    }

    downloadImages() {
        if (this.workflowState.image1 && this.workflowState.image2) {
            // Download annotated images
            this.downloadBase64Image(this.workflowState.image1.annotated_image, 'image1_detected.jpg');
            this.downloadBase64Image(this.workflowState.image2.annotated_image, 'image2_detected.jpg');
            this.showToast('Images downloaded successfully!', 'success');
        }
    }

    downloadBase64Image(base64Data, filename) {
        const link = document.createElement('a');
        link.download = filename;
        link.href = base64Data;
        link.click();
    }

    async resetWorkflow() {
        // Reset workflow state
        this.currentStep = 1;
        this.workflowState = {
            image1: null,
            image2: null,
            comparison: null
        };

        // Call backend reset
        try {
            await fetch('/reset-workflow', { method: 'POST' });
        } catch (error) {
            console.error('Reset error:', error);
        }

        // Hide all sections
        document.getElementById('individual-results').style.display = 'none';
        document.getElementById('both-images-summary').style.display = 'none';
        document.getElementById('comparison-section').style.display = 'none';
        document.getElementById('parameter-adjustment').style.display = 'none';
        document.getElementById('results-saving').style.display = 'none';

        // Reset file input
        document.getElementById('file-input').value = '';

        // Update UI
        this.updateWorkflowUI();

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });

        this.showToast('Workflow reset successfully!', 'success');
    }

    displayClassCounts(classCounts, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        // Sort by count (descending)
        const sortedCounts = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);

        sortedCounts.forEach(([className, count]) => {
            const item = document.createElement('div');
            item.className = 'class-count-item';
            item.innerHTML = `
                <span><strong>${className}</strong></span>
                <span class="badge">${count}</span>
            `;
            container.appendChild(item);
        });

        if (sortedCounts.length === 0) {
            container.innerHTML = '<p class="text-muted">No objects detected</p>';
        }
    }

    displayDetections(detections, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        // Sort by confidence (descending)
        const sortedDetections = detections.sort((a, b) => b.confidence - a.confidence);

        sortedDetections.forEach((detection, index) => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <div>
                    <strong>${detection.class_name}</strong> 
                    <span class="confidence">(${(detection.confidence * 100).toFixed(1)}%)</span>
                </div>
                <div class="detection-details">
                    Area: ${Math.round(detection.area)} pixels
                </div>
            `;
            container.appendChild(item);
        });

        if (sortedDetections.length === 0) {
            container.innerHTML = '<p class="text-muted">No objects detected</p>';
        }
    }

    displayClassChanges(classChanges) {
        const container = document.getElementById('class-changes-table');
        container.innerHTML = '';

        Object.entries(classChanges).forEach(([className, change]) => {
            const item = document.createElement('div');
            const changeValue = change.change;
            const changeClass = changeValue > 0 ? 'change-positive' : 
                               changeValue < 0 ? 'change-negative' : 'change-neutral';
            
            item.className = `change-item ${changeClass}`;
            item.innerHTML = `
                <span><strong>${className}</strong></span>
                <span>${change.image1} → ${change.image2} (${changeValue >= 0 ? '+' : ''}${changeValue})</span>
            `;
            container.appendChild(item);
        });
    }

    displayNewClasses(newClasses) {
        const container = document.getElementById('new-classes-list');
        container.innerHTML = '';

        if (newClasses.length > 0) {
            newClasses.forEach(className => {
                const item = document.createElement('div');
                item.className = 'class-item new-class';
                item.innerHTML = `<i class="fas fa-plus-circle"></i> ${className}`;
                container.appendChild(item);
            });
        } else {
            container.innerHTML = '<p class="text-muted">No new objects detected</p>';
        }
    }

    displayRemovedClasses(removedClasses) {
        const container = document.getElementById('removed-classes-list');
        container.innerHTML = '';

        if (removedClasses.length > 0) {
            removedClasses.forEach(className => {
                const item = document.createElement('div');
                item.className = 'class-item removed-class';
                item.innerHTML = `<i class="fas fa-minus-circle"></i> ${className}`;
                container.appendChild(item);
            });
        } else {
            container.innerHTML = '<p class="text-muted">No objects removed</p>';
        }
    }

    showLoading(show) {
        document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto-remove after 4 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 4000);
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    showMobileHTTPSHelp() {
        // Create and show a modal with HTTPS setup instructions for mobile
        const modal = document.createElement('div');
        modal.id = 'https-help-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            padding: 20px;
            box-sizing: border-box;
        `;
        
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            border-radius: 10px;
            padding: 25px;
            max-width: 500px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
        `;
        
        content.innerHTML = `
            <div style="text-align: right; margin-bottom: 15px;">
                <button id="https-help-close" style="background: none; border: none; font-size: 24px; cursor: pointer; padding: 0; color: #666;">&times;</button>
            </div>
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: #e74c3c; margin: 0;">📱 Mobile Camera Access Issue</h3>
            </div>
            <div style="text-align: left; line-height: 1.6;">
                <p><strong>Why this happens:</strong></p>
                <p>Mobile browsers require HTTPS (secure connection) to access the camera for security reasons. Your app is currently running over HTTP.</p>
                
                <p><strong>Solutions:</strong></p>
                <ol style="margin-left: 20px;">
                    <li><strong>Use File Upload Instead</strong> (Easiest)<br>
                        <span style="color: #666;">Tap the "Upload Image" button and select photos from your gallery</span></li>
                    
                    <li><strong>Enable HTTPS</strong> (Recommended)<br>
                        <span style="color: #666;">Ask the person running the server to enable HTTPS</span></li>
                    
                    <li><strong>Access via Computer</strong><br>
                        <span style="color: #666;">Use the camera feature on a desktop/laptop browser</span></li>
                </ol>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <strong>💡 Tip:</strong> The file upload option works great on mobile! You can take a photo with your phone's camera app and then upload it using the "Upload Image" button.
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button id="https-help-ok" style="background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 16px;">Got it!</button>
            </div>
        `;
        
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        // Close button handlers
        const closeModal = () => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        };
        
        document.getElementById('https-help-close').onclick = closeModal;
        document.getElementById('https-help-ok').onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ObjectDetectionApp();
});