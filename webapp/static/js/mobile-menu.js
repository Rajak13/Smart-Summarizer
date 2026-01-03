/**
 * Mobile Menu Functionality for Smart Summarizer
 * Handles responsive navigation menu toggle
 */

function toggleMobileMenu() {
    const navbarLinks = document.getElementById('navbarLinks');
    const menuToggle = document.querySelector('.mobile-menu-toggle i');
    
    if (!navbarLinks || !menuToggle) return;
    
    navbarLinks.classList.toggle('mobile-open');
    
    // Toggle hamburger/close icon
    if (navbarLinks.classList.contains('mobile-open')) {
        menuToggle.className = 'fas fa-times';
        // Prevent body scroll when menu is open
        document.body.style.overflow = 'hidden';
    } else {
        menuToggle.className = 'fas fa-bars';
        document.body.style.overflow = '';
    }
}

// Initialize mobile menu functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
        const navbar = document.querySelector('.top-navbar');
        const navbarLinks = document.getElementById('navbarLinks');
        const menuToggle = document.querySelector('.mobile-menu-toggle');
        
        if (!navbar || !navbarLinks || !menuToggle) return;
        
        if (!navbar.contains(event.target) && navbarLinks.classList.contains('mobile-open')) {
            navbarLinks.classList.remove('mobile-open');
            menuToggle.querySelector('i').className = 'fas fa-bars';
            document.body.style.overflow = '';
        }
    });

    // Close mobile menu when window is resized to desktop
    window.addEventListener('resize', function() {
        const navbarLinks = document.getElementById('navbarLinks');
        const menuToggle = document.querySelector('.mobile-menu-toggle i');
        
        if (!navbarLinks || !menuToggle) return;
        
        if (window.innerWidth > 768 && navbarLinks.classList.contains('mobile-open')) {
            navbarLinks.classList.remove('mobile-open');
            menuToggle.className = 'fas fa-bars';
            document.body.style.overflow = '';
        }
    });

    // Close mobile menu when clicking on nav items
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const navbarLinks = document.getElementById('navbarLinks');
            const menuToggle = document.querySelector('.mobile-menu-toggle i');
            
            if (navbarLinks && menuToggle && navbarLinks.classList.contains('mobile-open')) {
                navbarLinks.classList.remove('mobile-open');
                menuToggle.className = 'fas fa-bars';
                document.body.style.overflow = '';
            }
        });
    });
});

// Touch gesture support for mobile menu
let touchStartX = 0;
let touchEndX = 0;

document.addEventListener('touchstart', function(event) {
    touchStartX = event.changedTouches[0].screenX;
});

document.addEventListener('touchend', function(event) {
    touchEndX = event.changedTouches[0].screenX;
    handleSwipeGesture();
});

function handleSwipeGesture() {
    const navbarLinks = document.getElementById('navbarLinks');
    const menuToggle = document.querySelector('.mobile-menu-toggle i');
    
    if (!navbarLinks || !menuToggle) return;
    
    const swipeThreshold = 50;
    const swipeDistance = touchEndX - touchStartX;
    
    // Swipe right to open menu (only if menu is closed and swipe starts from left edge)
    if (swipeDistance > swipeThreshold && touchStartX < 50 && !navbarLinks.classList.contains('mobile-open')) {
        navbarLinks.classList.add('mobile-open');
        menuToggle.className = 'fas fa-times';
        document.body.style.overflow = 'hidden';
    }
    
    // Swipe left to close menu (only if menu is open)
    if (swipeDistance < -swipeThreshold && navbarLinks.classList.contains('mobile-open')) {
        navbarLinks.classList.remove('mobile-open');
        menuToggle.className = 'fas fa-bars';
        document.body.style.overflow = '';
    }
}

// Keyboard accessibility
document.addEventListener('keydown', function(event) {
    const navbarLinks = document.getElementById('navbarLinks');
    const menuToggle = document.querySelector('.mobile-menu-toggle');
    
    if (!navbarLinks || !menuToggle) return;
    
    // Close menu with Escape key
    if (event.key === 'Escape' && navbarLinks.classList.contains('mobile-open')) {
        navbarLinks.classList.remove('mobile-open');
        menuToggle.querySelector('i').className = 'fas fa-bars';
        document.body.style.overflow = '';
        menuToggle.focus(); // Return focus to menu button
    }
    
    // Toggle menu with Enter/Space when menu button is focused
    if ((event.key === 'Enter' || event.key === ' ') && event.target === menuToggle) {
        event.preventDefault();
        toggleMobileMenu();
    }
});