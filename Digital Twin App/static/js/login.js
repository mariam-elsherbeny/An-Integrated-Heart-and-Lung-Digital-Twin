async function login() {
const username = document.getElementById("doctor-id").value.trim();
const password = document.getElementById("password").value.trim();
const statusDiv = document.getElementById("login-status");

const response = await fetch("/api/doctor-login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, password: password })
});

const data = await response.json();

if (response.ok) {
    statusDiv.innerText = "Login successful.";
    window.location.href = `/patients?doctor_id=${data.doctor_id}`;
} else {
    statusDiv.innerText = data.error || "Login failed.";
}
}

// Add Enter key support for login
document.querySelectorAll("#doctor-id, #password").forEach(input => {
input.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
    event.preventDefault(); 
    login(); 
    }
});
});

