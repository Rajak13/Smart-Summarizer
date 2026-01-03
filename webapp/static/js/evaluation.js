// Evaluation Page - ROUGE Metrics Chart

// Sample benchmark data (from CNN/DailyMail evaluation)
const benchmarkData = {
    textrank: {
        rouge1: 0.43,
        rouge2: 0.18,
        rougeL: 0.35
    },
    bart: {
        rouge1: 0.51,
        rouge2: 0.34,
        rougeL: 0.48
    },
    pegasus: {
        rouge1: 0.55,
        rouge2: 0.30,
        rougeL: 0.52
    }
};

// Initialize chart
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('rougeChart').getContext('2d');
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['TextRank', 'BART', 'PEGASUS'],
            datasets: [
                {
                    label: 'ROUGE-1',
                    data: [
                        benchmarkData.textrank.rouge1,
                        benchmarkData.bart.rouge1,
                        benchmarkData.pegasus.rouge1
                    ],
                    backgroundColor: '#6D8196',
                    borderRadius: 6
                },
                {
                    label: 'ROUGE-2',
                    data: [
                        benchmarkData.textrank.rouge2,
                        benchmarkData.bart.rouge2,
                        benchmarkData.pegasus.rouge2
                    ],
                    backgroundColor: '#CBCBCB',
                    borderRadius: 6
                },
                {
                    label: 'ROUGE-L',
                    data: [
                        benchmarkData.textrank.rougeL,
                        benchmarkData.bart.rougeL,
                        benchmarkData.pegasus.rougeL
                    ],
                    backgroundColor: '#4A4A4A',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 12,
                            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'
                        },
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: '#4A4A4A',
                    padding: 12,
                    titleFont: {
                        size: 13
                    },
                    bodyFont: {
                        size: 12
                    },
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 0.6,
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6D8196'
                    },
                    grid: {
                        color: 'rgba(203, 203, 203, 0.3)',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        color: '#4A4A4A'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
});
