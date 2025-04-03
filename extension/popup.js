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
  
        // Handle AI result + confidence
        let message = "";
        const percent = data.confidence * 100;
  
        if (data.confidence < 0.6) {
          message = `🤔 <b>Uncertain</b><br/>Confidence: ${percent.toFixed(1)}%`;
        } else if (data.confidence < 0.85) {
          message = `⚠️ Possibly <b>${data.result.toUpperCase()}</b><br/>Confidence: ${percent.toFixed(1)}%`;
        } else {
          message = `❌ <b>${data.result.toUpperCase()}</b><br/>Confidence: ${percent.toFixed(1)}%`;
        }
  
        // If fact-check info exists, add it
        if (data.fact_check) {
          message += `
            <br/><br/><b>Fact Check Found:</b><br/>
            "${data.fact_check.text}"<br/>
            <b>Rating:</b> ${data.fact_check.rating}<br/>
            <a href="${data.fact_check.url}" target="_blank">View Source</a>
          `;
        }
  
        resultDiv.innerHTML = message;
      } catch (error) {
        resultDiv.textContent = "❌ Error connecting to AI server.";
        console.error(error);
      }
    });
  });
  