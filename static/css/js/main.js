// ScamSense — main.js

// FIX 3: Validate delay before using setTimeout
function autoHideAlert(elementId, delay) {
    const safeDelay = (typeof delay === 'number' && delay > 0) ? delay : 5000;
    setTimeout(() => {
        const el = document.getElementById(elementId);
        if (el) el.classList.add('d-none');
    }, safeDelay);
}

// FIX 2: Validate input before formatting
function formatINR(amount) {
    const num = parseFloat(amount);
    if (isNaN(num) || amount === null || amount === undefined) {
        return 'Rs.0';
    }
    return 'Rs.' + num.toLocaleString('en-IN');
}

function getCurrentTime() {
    return new Date().toLocaleTimeString();
}

// FIX 1: Use textContent instead of innerHTML to prevent XSS
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed bottom-0 end-0 m-3 shadow`;
    toast.style.zIndex    = '9999';
    toast.style.minWidth  = '250px';
    toast.textContent = message;   // safe — no XSS
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function confirmClear() {
    return confirm('Are you sure you want to clear all history?');
}