/* minimal: render Lucide icons */
window.addEventListener('DOMContentLoaded', () => {
  window.lucide && window.lucide.createIcons();
});

/* === Section Reveal on Scroll === */
const sections = document.querySelectorAll("section");

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      observer.unobserve(entry.target); // animate once
    }
  });
}, { threshold: 0.15 });

sections.forEach(section => {
  observer.observe(section);
});
