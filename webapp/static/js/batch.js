// Batch Processing Page

let batchQueue = [];
let batchResults = [];

// DOM Elements
const loadSamplesBtn = document.getElementById('loadSamplesBtn');
const runBatchBtn = document.getElementById('runBatchBtn');
const exportBtn = document.getElementById('exportBtn');
const tableBody = document.getElementById('batchTableBody');

// Sample texts for demo
const sampleTexts = [
    {
        text: "Artificial intelligence has revolutionized the way we interact with technology. Machine learning algorithms can now process vast amounts of data and identify patterns that humans might miss. Deep learning neural networks have enabled breakthroughs in computer vision, natural language processing, and speech recognition. These advances are transforming industries from healthcare to finance.",
        models: ['textrank', 'bart', 'pegasus']
    },
    {
        text: "Climate change poses one of the greatest challenges to humanity. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. Scientists warn that without immediate action, the consequences could be catastrophic for future generations.",
        models: ['textrank', 'bart']
    },
    {
        text: "The human brain is the most complex organ in the body, containing approximately 86 billion neurons. These neurons communicate through electrical and chemical signals, forming intricate networks that enable thought, memory, and consciousness. Neuroscientists continue to uncover the mysteries of how the brain processes information and generates our subjective experiences.",
        models: ['bart', 'pegasus']
    }
];

// Load sample documents
loadSamplesBtn.addEventListener('click', function() {
    batchQueue = [...sampleTexts];
    renderTable();
});

// Run batch processing
runBatchBtn.addEventListener('click', async function() {
    if (batchQueue.length === 0) {
        alert('No items in queue. Please load samples first.');
        return;
    }
    
    runBatchBtn.disabled = true;
    runBatchBtn.textContent = 'Processing...';
    
    for (let i = 0; i < batchQueue.length; i++) {
        const item = batchQueue[i];
        item.status = 'processing';
        renderTable();
        
        try {
            const results = {};
            
            for (const model of item.models) {
                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: item.text,
                        model: model
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    results[model] = {
                        summary: data.summary,
                        metadata: data.metadata
                    };
                }
            }
            
            item.results = results;
            item.status = 'complete';
            batchResults.push(item);
            
        } catch (error) {
            item.status = 'error';
            item.error = error.message;
        }
        
        renderTable();
    }
    
    runBatchBtn.disabled = false;
    runBatchBtn.textContent = 'Run Batch';
});

// Export results to CSV
exportBtn.addEventListener('click', function() {
    if (batchResults.length === 0) {
        alert('No results to export. Please run batch processing first.');
        return;
    }
    
    let csv = 'Source Text,Model,Summary,Processing Time (s),Compression Ratio\n';
    
    batchResults.forEach(item => {
        if (item.results) {
            Object.keys(item.results).forEach(model => {
                const result = item.results[model];
                const sourceText = item.text.replace(/"/g, '""').substring(0, 100) + '...';
                const summary = result.summary.replace(/"/g, '""');
                const time = result.metadata.processing_time.toFixed(2);
                const compression = (result.metadata.compression_ratio * 100).toFixed(1) + '%';
                
                csv += `"${sourceText}","${model}","${summary}",${time},${compression}\n`;
            });
        }
    });
    
    // Download CSV
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_results_' + new Date().toISOString().split('T')[0] + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
});

// Render table
function renderTable() {
    if (batchQueue.length === 0) {
        tableBody.innerHTML = `
            <tr class="empty-state">
                <td colspan="4">
                    <div class="empty-message">
                        No items in the queue. Load samples or upload a CSV to begin.
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    tableBody.innerHTML = batchQueue.map((item, index) => {
        const preview = item.text.substring(0, 80) + '...';
        const modelBadges = item.models.map(m => 
            `<span class="model-badge">${m.toUpperCase()}</span>`
        ).join('');
        
        let statusBadge = '';
        if (!item.status || item.status === 'pending') {
            statusBadge = '<span class="status-badge status-pending">Pending</span>';
        } else if (item.status === 'processing') {
            statusBadge = '<span class="status-badge status-processing">Processing...</span>';
        } else if (item.status === 'complete') {
            statusBadge = '<span class="status-badge status-complete">Complete</span>';
        } else if (item.status === 'error') {
            statusBadge = '<span class="status-badge status-error">Error</span>';
        }
        
        return `
            <tr>
                <td><div class="source-preview">${preview}</div></td>
                <td><div class="model-badges">${modelBadges}</div></td>
                <td>${statusBadge}</td>
                <td>
                    <div class="action-buttons">
                        <button class="btn-icon" onclick="viewItem(${index})" ${item.status !== 'complete' ? 'disabled' : ''}>View</button>
                        <button class="btn-icon" onclick="removeItem(${index})">Remove</button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
}

// View item results
function viewItem(index) {
    const item = batchQueue[index];
    if (!item.results) return;
    
    let resultsHtml = '<div style="max-width: 800px; margin: 0 auto;">';
    resultsHtml += '<h3 style="margin-bottom: 1rem;">Batch Results</h3>';
    resultsHtml += `<p style="color: #6D8196; margin-bottom: 2rem;"><strong>Source:</strong> ${item.text.substring(0, 200)}...</p>`;
    
    Object.keys(item.results).forEach(model => {
        const result = item.results[model];
        resultsHtml += `
            <div style="margin-bottom: 2rem; padding: 1.5rem; background: #F5F0F6; border-radius: 8px;">
                <h4 style="margin-bottom: 0.5rem; color: #4A4A4A;">${model.toUpperCase()}</h4>
                <p style="line-height: 1.8; margin-bottom: 1rem;">${result.summary}</p>
                <div style="display: flex; gap: 2rem; font-size: 0.9rem; color: #6D8196;">
                    <span><strong>Time:</strong> ${result.metadata.processing_time.toFixed(2)}s</span>
                    <span><strong>Compression:</strong> ${(result.metadata.compression_ratio * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
    });
    
    resultsHtml += '</div>';
    
    // Create modal
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 9999; padding: 2rem;';
    modal.innerHTML = `
        <div style="background: white; border-radius: 12px; padding: 2rem; max-height: 90vh; overflow-y: auto; position: relative;">
            <button onclick="this.parentElement.parentElement.remove()" style="position: absolute; top: 1rem; right: 1rem; background: none; border: none; font-size: 1.5rem; cursor: pointer; color: #4A4A4A;">×</button>
            ${resultsHtml}
        </div>
    `;
    document.body.appendChild(modal);
}

// Remove item from queue
function removeItem(index) {
    batchQueue.splice(index, 1);
    renderTable();
}

// Initial render
renderTable();
