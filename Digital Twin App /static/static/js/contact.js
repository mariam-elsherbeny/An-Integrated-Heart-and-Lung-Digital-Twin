document.addEventListener("DOMContentLoaded", () => {
    const contactForm = document.getElementById("contact-form");

    if (!contactForm) {
        console.error("Contact form not found! ");
        return;
    }

    contactForm.addEventListener("submit", (event) => {
        event.preventDefault();

        const patientId = document.getElementById("patient-id")?.value.trim();
        const email = document.getElementById("email")?.value.trim();
        const message = document.getElementById("message")?.value.trim();

        if (!patientId || !email || !message) {
            alert("❌ Please fill in all fields, darling.");
            return;
        }

        alert("✅ Submission successful! We have received your message, gorgeous.");
        contactForm.reset();
    });
});
