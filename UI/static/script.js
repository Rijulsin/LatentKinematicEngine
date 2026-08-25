document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('retrieval-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    
    // Result elements
    const confidenceBadge = document.getElementById('confidence-badge');
    const timelineHighlight = document.getElementById('timeline-highlight');
    const durationLabel = document.getElementById('duration-label');
    const startTimeEl = document.getElementById('start-time');
    const endTimeEl = document.getElementById('end-time');
    const extractedNoun = document.getElementById('extracted-noun');
    const extractedVerb = document.getElementById('extracted-verb');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const videoId = document.getElementById('video-id').value.trim();
        const query = document.getElementById('query').value.trim();
        
        if (!videoId || !query) return;

        // Reset UI state
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');
        timelineHighlight.style.width = '0%';
        timelineHighlight.style.left = '0%';

        try {
            const response = await fetch('/api/retrieve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ video_id: videoId, query: query }),
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Failed to retrieve moment');
            }

            // Update UI with results
            confidenceBadge.textContent = `${data.confidence.toFixed(1)}% Confidence`;
            
            // Set colors based on confidence
            if (data.confidence > 80) {
                confidenceBadge.style.color = 'var(--success)';
                confidenceBadge.style.background = 'rgba(16, 185, 129, 0.2)';
                confidenceBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            } else if (data.confidence > 50) {
                confidenceBadge.style.color = '#f59e0b'; // warning orange
                confidenceBadge.style.background = 'rgba(245, 158, 11, 0.2)';
                confidenceBadge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
            } else {
                confidenceBadge.style.color = 'var(--error)';
                confidenceBadge.style.background = 'rgba(239, 68, 68, 0.2)';
                confidenceBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
            }

            startTimeEl.textContent = `${data.start_time.toFixed(1)}s`;
            endTimeEl.textContent = `${data.end_time.toFixed(1)}s`;
            durationLabel.textContent = `${data.video_duration.toFixed(1)}s`;
            extractedNoun.textContent = data.noun_extracted;
            extractedVerb.textContent = data.verb_extracted;

            // Animate timeline
            const startPercent = (data.start_time / data.video_duration) * 100;
            const endPercent = (data.end_time / data.video_duration) * 100;
            const widthPercent = endPercent - startPercent;

            resultContainer.classList.remove('hidden');
            
            // Small delay to allow display:block to apply before animating width
            setTimeout(() => {
                timelineHighlight.style.left = `${startPercent}%`;
                timelineHighlight.style.width = `${widthPercent}%`;
            }, 50);

        } catch (error) {
            errorText.textContent = error.message;
            errorMessage.classList.remove('hidden');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });
});
