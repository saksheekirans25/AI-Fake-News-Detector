document.addEventListener("DOMContentLoaded", () => {
    const checkBtn = document.getElementById("checkBtn");
    const resultDiv = document.getElementById("result");
  
    checkBtn.addEventListener("click", async () => {
      const text = document.getElementById("newsInput").value;
  
      if (!text.trim()) {
        resultDiv.textContent = "Please enter some text.";
        return;
      }
  
      resultDiv.textContent = "Checking...";
  
      try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
  
        const data = await response.json();
        resultDiv.innerHTML = `Result: <b>${data.result.toUpperCase()}</b><br/>Confidence: ${data.confidence * 100}%`;
      } catch (error) {
        resultDiv.textContent = "Error connecting to AI server.";
      }
    });
  });
  