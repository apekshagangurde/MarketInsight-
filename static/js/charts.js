// Wait for the DOM to be loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get the forecast data if available
    const forecastDataElement = document.getElementById('forecast-data');
    
    if (forecastDataElement) {
        const forecastData = JSON.parse(forecastDataElement.getAttribute('data-forecast'));
        
        // Create forecast chart
        createForecastChart(forecastData);
    }
    
    // Add event listeners to stock suggestion elements
    const stockSuggestions = document.querySelectorAll('.stock-suggestion');
    stockSuggestions.forEach(suggestion => {
        suggestion.addEventListener('click', function() {
            document.getElementById('stock_symbol').value = this.getAttribute('data-symbol');
        });
    });
    
    // Add form submission loading indicator
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function() {
            // Show loading spinner
            const spinnerHTML = `
                <div class="spinner-container" id="loading-spinner">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', spinnerHTML);
        });
    }
});

// Function to create forecast chart
function createForecastChart(forecastData) {
    // Get the canvas element
    const ctx = document.getElementById('forecast-chart').getContext('2d');
    
    // Create labels for the next 7 days
    const labels = [];
    const currentDate = new Date();
    
    for (let i = 0; i < forecastData.length; i++) {
        const nextDate = new Date();
        nextDate.setDate(currentDate.getDate() + i + 1);
        labels.push(nextDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    
    // Create the chart
    const forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price Forecast',
                data: forecastData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '7-Day Price Forecast',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Price'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

// Function to create comparison chart for model predictions
function createComparisonChart() {
    const ctx = document.getElementById('comparison-chart').getContext('2d');
    
    // Get prediction values from HTML
    const lastPrice = parseFloat(document.getElementById('last-price').textContent);
    const arimaPrice = parseFloat(document.getElementById('arima-pred').textContent);
    const lstmPrice = parseFloat(document.getElementById('lstm-pred').textContent);
    const linRegPrice = parseFloat(document.getElementById('lin-reg-pred').textContent);
    const weightedPrice = parseFloat(document.getElementById('weighted-pred').textContent);
    
    const comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Last Closing', 'ARIMA', 'LSTM', 'Linear Regression', 'Weighted Average'],
            datasets: [{
                label: 'Stock Price Predictions',
                data: [lastPrice, arimaPrice, lstmPrice, linRegPrice, weightedPrice],
                backgroundColor: [
                    'rgba(200, 200, 200, 0.6)',
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(75, 192, 192, 0.6)'
                ],
                borderColor: [
                    'rgba(200, 200, 200, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Predictions Comparison',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Call comparison chart when page loads
if (document.getElementById('comparison-chart')) {
    createComparisonChart();
}
