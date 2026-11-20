/**
 * F1 Race Predictor - Client-side Interaction Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const driverSelect = document.getElementById("driverSelect");
    const constructorSelect = document.getElementById("constructorSelect");
    const circuitSelect = document.getElementById("circuitSelect");
    const gridInput = document.getElementById("gridInput");
    const qualiInput = document.getElementById("qualiInput");

    const driverFormSlider = document.getElementById("driverFormSlider");
    const driverFormVal = document.getElementById("driverFormVal");
    const teamPerfSlider = document.getElementById("teamPerfSlider");
    const teamPerfVal = document.getElementById("teamPerfVal");

    const trackAltitude = document.getElementById("trackAltitude");
    const trackRaces = document.getElementById("trackRaces");
    const driverPodiumsStat = document.getElementById("driverPodiumsStat");
    const driverStartsStat = document.getElementById("driverStartsStat");

    const predictBtn = document.getElementById("predictBtn");
    const resultCard = document.getElementById("resultCard");
    const errorAlert = document.getElementById("errorAlert");

    // Slider Live Listeners
    if (driverFormSlider && driverFormVal) {
        driverFormSlider.addEventListener("input", (e) => {
            driverFormVal.textContent = parseFloat(e.target.value).toFixed(1) + " / 10";
        });
    }

    if (teamPerfSlider && teamPerfVal) {
        teamPerfSlider.addEventListener("input", (e) => {
            teamPerfVal.textContent = parseFloat(e.target.value).toFixed(1) + " / 10";
        });
    }

    // Grid to Quali Auto-sync
    if (gridInput && qualiInput) {
        gridInput.addEventListener("input", (e) => {
            if (!qualiInput.dataset.userModified) {
                qualiInput.value = e.target.value;
            }
        });
        qualiInput.addEventListener("input", () => {
            qualiInput.dataset.userModified = "true";
        });
    }

    // Dynamic Driver Stats Fetch
    if (driverSelect) {
        driverSelect.addEventListener("change", async (e) => {
            const driverName = e.target.value;
            if (!driverName) return;

            try {
                const res = await fetch(`/api/driver-stats/${encodeURIComponent(driverName)}`);
                const data = await res.json();
                if (data.success && data.stats) {
                    const s = data.stats;
                    // Update form slider
                    if (s.driver_recent_form_score !== undefined && driverFormSlider) {
                        driverFormSlider.value = s.driver_recent_form_score;
                        driverFormVal.textContent = parseFloat(s.driver_recent_form_score).toFixed(1) + " / 10";
                    }
                    // Update stats pills
                    if (driverPodiumsStat) driverPodiumsStat.textContent = s.driver_career_podiums || "0";
                    if (driverStartsStat) driverStartsStat.textContent = s.driver_career_starts || "0";
                    
                    // Auto-select last constructor if available
                    if (s.last_constructor && constructorSelect) {
                        for (let opt of constructorSelect.options) {
                            if (opt.value.toLowerCase() === s.last_constructor.toLowerCase()) {
                                constructorSelect.value = opt.value;
                                triggerConstructorUpdate(opt.value);
                                break;
                            }
                        }
                    }
                }
            } catch (err) {
                console.error("Error fetching driver stats:", err);
            }
        });
    }

    // Dynamic Constructor Stats Fetch
    function triggerConstructorUpdate(constructorName) {
        if (!constructorName) return;
        fetch(`/api/constructor-stats/${encodeURIComponent(constructorName)}`)
            .then(r => r.json())
            .then(data => {
                if (data.success && data.stats) {
                    const s = data.stats;
                    if (s.constructor_recent_performance_score !== undefined && teamPerfSlider) {
                        teamPerfSlider.value = s.constructor_recent_performance_score;
                        teamPerfVal.textContent = parseFloat(s.constructor_recent_performance_score).toFixed(1) + " / 10";
                    }
                }
            })
            .catch(err => console.error("Error fetching constructor stats:", err));
    }

    if (constructorSelect) {
        constructorSelect.addEventListener("change", (e) => {
            triggerConstructorUpdate(e.target.value);
        });
    }

    // Dynamic Circuit Stats Fetch
    if (circuitSelect) {
        circuitSelect.addEventListener("change", async (e) => {
            const circuitName = e.target.value;
            if (!circuitName) return;

            try {
                const res = await fetch(`/api/circuit-stats/${encodeURIComponent(circuitName)}`);
                const data = await res.json();
                if (data.success && data.stats) {
                    const s = data.stats;
                    if (trackAltitude) trackAltitude.textContent = (s.altitude || 0) + " m";
                    if (trackRaces) trackRaces.textContent = (s.total_races || 0) + " Grands Prix";
                }
            } catch (err) {
                console.error("Error fetching circuit stats:", err);
            }
        });
    }

    // Prediction Submit Handler
    if (predictBtn) {
        predictBtn.addEventListener("click", async () => {
            const driver = driverSelect ? driverSelect.value : "";
            const constructor = constructorSelect ? constructorSelect.value : "";
            const circuit = circuitSelect ? circuitSelect.value : "";
            const grid = gridInput ? parseInt(gridInput.value) : 1;
            const quali = qualiInput ? parseInt(qualiInput.value) : grid;
            const driverForm = driverFormSlider ? parseFloat(driverFormSlider.value) : 5.0;
            const teamPerf = teamPerfSlider ? parseFloat(teamPerfSlider.value) : 5.0;

            if (!driver || !constructor || !circuit) {
                showError("Please select a valid Driver, Constructor, and Track.");
                return;
            }

            if (isNaN(grid) || grid < 1 || grid > 30) {
                showError("Grid position must be a valid integer between 1 and 30.");
                return;
            }

            // UI Loading state
            predictBtn.disabled = true;
            predictBtn.innerHTML = `<span>⚙️ CALCULATING TELEMETRY...</span>`;
            hideError();

            try {
                const res = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        driver: driver,
                        constructor: constructor,
                        circuit: circuit,
                        grid_position: grid,
                        qualifying_position: quali,
                        driver_form_score: driverForm,
                        constructor_perf_score: teamPerf,
                    })
                });

                const json = await res.json();

                if (!json.success) {
                    showError(json.error || "Prediction calculation failed.");
                } else {
                    displayPredictionResults(json.data);
                }
            } catch (err) {
                showError("Network error connecting to prediction engine.");
            } finally {
                predictBtn.disabled = false;
                predictBtn.innerHTML = `<span>🏁 PREDICT PODIUM</span>`;
            }
        });
    }

    function displayPredictionResults(data) {
        if (!resultCard) return;

        // Elements in result card
        const resDriver = document.getElementById("resDriver");
        const resTeam = document.getElementById("resTeam");
        const resTrack = document.getElementById("resTrack");
        const resBadge = document.getElementById("resBadge");
        const resProbPct = document.getElementById("resProbPct");
        const resProbFill = document.getElementById("resProbFill");
        const resConfidence = document.getElementById("resConfidence");
        const resFactorsList = document.getElementById("resFactorsList");
        const driverAvatar = document.getElementById("driverAvatar");

        if (resDriver) resDriver.textContent = data.driver;
        if (resTeam) resTeam.textContent = data.constructor;
        if (resTrack) resTrack.textContent = data.circuit + " (Grid: P" + data.grid_position + ")";
        if (driverAvatar) {
            const initials = data.driver.split(" ").map(n => n[0]).join("");
            driverAvatar.textContent = initials;
        }

        // Outcome Badge
        if (resBadge) {
            if (data.is_podium) {
                resBadge.className = "badge-outcome podium";
                resBadge.textContent = "🏆 PODIUM FINISH";
            } else {
                resBadge.className = "badge-outcome no-podium";
                resBadge.textContent = "❌ NO PODIUM";
            }
        }

        // Probability meter
        if (resProbPct) resProbPct.textContent = data.podium_probability + "%";
        if (resProbFill) {
            resProbFill.style.width = data.podium_probability + "%";
            if (data.is_podium) {
                resProbFill.className = "progress-fill gold";
            } else {
                resProbFill.className = "progress-fill";
            }
        }

        // Confidence
        if (resConfidence) resConfidence.textContent = data.confidence + "%";

        // Factors List
        if (resFactorsList) {
            resFactorsList.innerHTML = "";
            if (data.explanation_factors && data.explanation_factors.length > 0) {
                data.explanation_factors.forEach(f => {
                    const item = document.createElement("div");
                    item.className = `factor-item ${f.type}`;
                    const icon = f.type === "positive" ? "✓" : (f.type === "negative" ? "✗" : "•");
                    item.innerHTML = `<span class="factor-icon">${icon}</span> <span>${f.text}</span>`;
                    resFactorsList.appendChild(item);
                });
            }
        }

        // Show card and scroll
        resultCard.classList.add("show");
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function showError(msg) {
        if (errorAlert) {
            errorAlert.textContent = msg;
            errorAlert.style.display = "block";
        } else {
            alert(msg);
        }
    }

    function hideError() {
        if (errorAlert) errorAlert.style.display = "none";
    }

    // Initial Trigger for default selections
    if (driverSelect && driverSelect.value) {
        driverSelect.dispatchEvent(new Event("change"));
    }
    if (circuitSelect && circuitSelect.value) {
        circuitSelect.dispatchEvent(new Event("change"));
    }
});
