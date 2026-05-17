// static/js/app.js
document.addEventListener('DOMContentLoaded', function () {
    // 1. MANAJEMEN PREFERENSI GLOBAL UNIVERSAL DARK MODE
    const themeToggler = document.getElementById('themeToggler');
    const themeIcon = document.getElementById('themeIcon');
    const htmlElement = document.documentElement;

    const savedTheme = localStorage.getItem('theme') || 'light';
    htmlElement.setAttribute('data-bs-theme', savedTheme);
    updateTogglerIcon(savedTheme);

    if (themeToggler) {
        themeToggler.addEventListener('click', () => {
            const currentTheme = htmlElement.getAttribute('data-bs-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            htmlElement.setAttribute('data-bs-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateTogglerIcon(newTheme);
        });
    }

    function updateTogglerIcon(theme) {
        if (!themeIcon) return;
        if (theme === 'dark') {
            themeIcon.className = 'fa-solid fa-sun text-warning';
        } else {
            themeIcon.className = 'fa-solid fa-moon text-secondary';
        }
    }

    // 2. BOOTSTRAP 5 CLIENT VALIDATION & ASYNC LOADING OVERLAY
    const predForm = document.getElementById('predictionForm');
    const globalLoader = document.getElementById('globalLoader');

    if (predForm) {
        predForm.addEventListener('submit', function (event) {
            if (!predForm.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                predForm.classList.add('was-validated');
            } else {
                // Tampilkan loading screen jika data valid dan siap di-submit ke backend
                if (globalLoader) {
                    globalLoader.style.display = 'flex';
                }
            }
        }, false);
    }
});