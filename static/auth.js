(() => {
  const demoSessionKey = "knowledgeai.demo.session.v1";
  const form = document.querySelector("#login-form");
  const notice = document.querySelector("#login-notice");
  const password = document.querySelector("#login-password");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = document.querySelector("#login-email").value.trim();
    const displayName = email.split("@")[0]
      .replace(/[._-]+/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());

    try {
      sessionStorage.setItem(demoSessionKey, displayName);
      notice.textContent = "";
    } catch (error) {
      console.warn("Could not save the demo session.", error);
      notice.textContent = "Browser storage is unavailable; this demo session will not persist.";
    }

    window.dispatchEvent(new CustomEvent("knowledgeai:demo-login", {
      detail: { displayName },
    }));
  });

  document.querySelector("#password-toggle").addEventListener("click", (event) => {
    const button = event.currentTarget;
    const isVisible = password.type === "text";
    password.type = isVisible ? "password" : "text";
    button.textContent = isVisible ? "Show" : "Hide";
    button.setAttribute("aria-label", isVisible ? "Show password" : "Hide password");
  });

  document.querySelector("#signout-button").addEventListener("click", () => {
    try {
      sessionStorage.removeItem(demoSessionKey);
    } catch (error) {
      console.warn("Could not clear the demo session.", error);
    }
    window.dispatchEvent(new Event("knowledgeai:demo-logout"));
  });

  try {
    const displayName = sessionStorage.getItem(demoSessionKey);
    if (displayName) {
      window.dispatchEvent(new CustomEvent("knowledgeai:demo-login", {
        detail: { displayName },
      }));
    }
  } catch (error) {
    console.warn("Could not read the demo session.", error);
    notice.textContent = "Browser storage is unavailable; sign in again after reloading.";
  }
})();
