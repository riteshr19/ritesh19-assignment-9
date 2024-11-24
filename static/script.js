document.getElementById("experiment-form").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent form submission

    const activation = document.getElementById("activation").value;
    const lr = parseFloat(document.getElementById("lr").value);
    const hiddenSize = parseInt(document.getElementById("hidden_size").value);
    const epochs = parseInt(document.getElementById("epochs").value);

    // Validation checks
    const acts = ["relu", "tanh", "sigmoid"];
    if (!acts.includes(activation)) {
        alert("Please choose from relu, tanh, sigmoid.");
        return;
    }

    if (isNaN(lr)) {
        alert("Please enter a valid number for learning rate.");
        return;
    }

    if (isNaN(hiddenSize) || hiddenSize <= 0) {
        alert("Please enter a positive integer for Hidden Layer Size.");
        return;
    }

    if (isNaN(epochs) || epochs <= 0) {
        alert("Please enter a positive integer for Number of Epochs.");
        return;
    }

    // If all validations pass, submit the form
    fetch("/run_experiment", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ activation: activation, lr: lr, hidden_size: hiddenSize, epochs: epochs })
    })
    .then(response => response.json())
    .then(data => {
        // Show and set images if they exist
        const resultsDiv = document.getElementById("results");
        resultsDiv.style.display = "block";

        const resultImg = document.getElementById("result_gif");
        if (data.result_gif) {
            resultImg.src = `/${data.result_gif}`;
            resultImg.style.display = "block";
        }
    })
    .catch(error => {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment.");
    });
});

// Add event listener for the Random button
document.getElementById("random-button").addEventListener("click", function() {
    const activations = ["relu", "tanh", "sigmoid"];
    const randomActivation = activations[Math.floor(Math.random() * activations.length)];
    const randomLr = (Math.random() * 0.1).toFixed(2);  // Random learning rate between 0 and 0.1
    const randomHiddenSize = Math.floor(Math.random() * 10) + 1;  // Random hidden size between 1 and 10
    const randomEpochs = Math.floor(Math.random() * 1000) + 1;  // Random epochs between 1 and 1000

    document.getElementById("activation").value = randomActivation;
    document.getElementById("lr").value = randomLr;
    document.getElementById("hidden_size").value = randomHiddenSize;
    document.getElementById("epochs").value = randomEpochs;
});

// Add event listener for the Clear button
document.getElementById("clear-button").addEventListener("click", function() {
    document.getElementById("activation").value = "";
    document.getElementById("lr").value = "";
    document.getElementById("hidden_size").value = "";
    document.getElementById("epochs").value = "";
    document.getElementById("results").style.display = "none";
    document.getElementById("result_gif").style.display = "none";
});