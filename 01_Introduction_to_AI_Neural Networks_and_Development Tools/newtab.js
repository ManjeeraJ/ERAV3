function getRandomQuote() {
    const randomIndex = Math.floor(Math.random() * quotes.length);
    return quotes[randomIndex];
}

function displayQuote() {
    const quoteData = getRandomQuote();
    document.getElementById('quote').textContent = quoteData.text;
    document.getElementById('attribution').textContent = `- ${quoteData.attribution}`;
}

// Display quote when page loads
document.addEventListener('DOMContentLoaded', displayQuote); 