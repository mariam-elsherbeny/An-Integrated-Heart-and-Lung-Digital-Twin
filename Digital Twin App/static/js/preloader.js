document.addEventListener('DOMContentLoaded', () => {
  const preloader     = document.getElementById('preloader');
  const welcome       = document.getElementById('welcome');
  const percentageEl  = document.getElementById('loading-percentage');
  const ecgAudio      = document.getElementById('ecgAudio');
  const welcomeAudio  = document.getElementById('welcomeAudio');

  let pct = 0;
  let userInteracted = false;
  let hasPlayedWelcome = false;

  // Ensure audio can play after user interaction (required by most browsers)
  document.body.addEventListener('click', () => {
    if (userInteracted) return;
    userInteracted = true;

    ecgAudio.currentTime = 0;
    ecgAudio.loop = true;
    ecgAudio.volume = 0.4;
    ecgAudio.play().catch(err => {
      console.warn('ECG audio playback failed:', err);
    });

    if (!hasPlayedWelcome) {
      hasPlayedWelcome = true;
      welcomeAudio.volume = 0.6;
      welcomeAudio.play().catch(err => {
        console.warn('Welcome audio playback failed:', err);
      });
    }
  }, { once: true });

  // Simulate loading percentage
  const timer = setInterval(() => {
    pct += 2;
    if (pct > 100) pct = 100;
    percentageEl.textContent = `${pct}%`;

    if (pct === 100) {
      clearInterval(timer);
      preloader.classList.add('hide');

      setTimeout(() => {
        ecgAudio.pause();
        ecgAudio.currentTime = 0;
        preloader.remove();

        document.body.classList.add('show-welcome');

        if (!hasPlayedWelcome) {
          hasPlayedWelcome = true;
          welcomeAudio.currentTime = 0;
          welcomeAudio.volume = 0.6;
          welcomeAudio.play().catch(() => {
            // fallback to user click
          });
        }

        setTimeout(() => {
          document.body.classList.remove('show-welcome');
          document.body.classList.add('loaded');
        }, 2000);

      }, 700);
    }
  }, 60);
});
