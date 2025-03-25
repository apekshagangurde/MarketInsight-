// Main JavaScript for stock prediction app

document.addEventListener('DOMContentLoaded', function() {
    // Form submission handling with loading state
    const stockForm = document.getElementById('stock-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    if (stockForm) {
        stockForm.addEventListener('submit', function() {
            loadingSpinner.style.display = 'block';
        });
    }
    
    // Chart initialization if we're on the prediction page
    const stockChartElement = document.getElementById('stockChart');
    if (stockChartElement) {
        initializeStockChart();
    }
    
    // Tooltip initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

function initializeStockChart() {
    // Fetch the stock data from the server
    fetch('/stock-data')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('stockChart').getContext('2d');
            
            // Create the chart
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [{
                        label: 'Stock Price',
                        data: data.prices,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: {
                                display: false
                            }
                        },
                        y: {
                            beginAtZero: false
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                        },
                        legend: {
                            position: 'top',
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching stock data:', error));
}

function toggleModelDetails(modelId) {
    const detailElement = document.getElementById(`${modelId}-details`);
    if (detailElement) {
        detailElement.classList.toggle('d-none');
    }
}
