
    const $ = (id) => document.getElementById(id);

    let AIRPORTS_DB = {
      MAD: { name: "Madrid Barajas", city: "Madrid", lat: 40.4722, lon: -3.5608 },
      TFS: { name: "Tenerife Sur", city: "Tenerife", lat: 28.0445, lon: -16.5725 },
      TFN: { name: "Tenerife Norte", city: "Tenerife", lat: 28.4827, lon: -16.3415 },
      BCN: { name: "Barcelona", city: "Barcelona", lat: 41.2971, lon: 2.0785 },
      JFK: { name: "New York", city: "New York", lat: 40.6413, lon: -73.7781 },
      LHR: { name: "Londres Heathrow", city: "Londres", lat: 51.4700, lon: -0.4543 },
      CDG: { name: "París Charles de Gaulle", city: "París", lat: 49.0097, lon: 2.5479 },
      DXB: { name: "Dubai", city: "Dubai", lat: 25.2532, lon: 55.3657 },
      LPA: { name: "Gran Canaria", city: "Gran Canaria", lat: 27.9319, lon: -15.3866 },
      PMI: { name: "Mallorca", city: "Palma", lat: 39.5517, lon: 2.7388 },
      AGP: { name: "Málaga", city: "Málaga", lat: 36.6749, lon: -4.4991 },
      AMS: { name: "Ámsterdam", city: "Ámsterdam", lat: 52.3086, lon: 4.7639 }
    };

    let map, dotsLayer, mainChart;
    let currentOrig = "MAD", currentDest = "TFS";

    async function loadAirports() {
      try {
        const res = await fetch('https://raw.githubusercontent.com/jbrooksuk/JSON-Airports/master/airports.json');
        const data = await res.json();
        for (let x of data) {
          if (x.iata && x.iata !== "\\N") {
            AIRPORTS_DB[x.iata] = { name: x.name, city: x.city, lat: parseFloat(x.lat), lon: parseFloat(x.lon) };
          }
        }
      } catch (e) {
        console.error("Fallback a VIP DB", e);
      }
      renderInitialInputs();
      fetchFlights();
    }

    function renderInitialInputs() {
      $('origInput').value = `${AIRPORTS_DB["MAD"]?.name || "Madrid Barajas"} (MAD)`;
      $('destInput').value = `${AIRPORTS_DB["TFS"]?.name || "Tenerife Sur"} (TFS)`;
    }

    function showSuggestions(type) {
      const input = $(type + 'Input');
      const val = input.value.toLowerCase().trim();
      const box = $(type + 'Suggestions');
      box.innerHTML = '';

      if (val.length < 1) {
        box.style.display = 'none';
        return;
      }

      let matches = [];
      for (const [iata, d] of Object.entries(AIRPORTS_DB)) {
        if (iata.toLowerCase().includes(val) || (d.name || '').toLowerCase().includes(val) || (d.city || '').toLowerCase().includes(val)) {
          matches.push({ iata, ...d });
        }
      }

      matches.sort((a, b) => {
        if (a.iata.toLowerCase() === val) return -1;
        if (b.iata.toLowerCase() === val) return 1;
        return 0;
      });

      matches = matches.slice(0, 10);

      if (matches.length > 0) {
        matches.forEach(m => {
          const div = document.createElement('div');
          div.className = 'suggestion-item';
          div.innerHTML = `<b>${m.iata}</b> - ${m.name} <small>(${m.city})</small>`;
          div.onmousedown = (e) => {
            e.preventDefault();
            input.value = `${m.name} (${m.iata})`;
            if (type === 'orig') currentOrig = m.iata;
            if (type === 'dest') currentDest = m.iata;
            box.style.display = 'none';
            fetchFlights();
          };
          div.ontouchstart = (e) => {
            e.preventDefault();
            input.value = `${m.name} (${m.iata})`;
            if (type === 'orig') currentOrig = m.iata;
            if (type === 'dest') currentDest = m.iata;
            box.style.display = 'none';
            fetchFlights();
          };
          box.appendChild(div);
        });
        box.style.display = 'block';
      } else {
        box.style.display = 'none';
      }
    }

    function hideSuggestions(type) {
      setTimeout(() => {
        $(type + 'Suggestions').style.display = 'none';
        const val = $(type + 'Input').value.trim();
        const fallbackMatch = val.match(/\(([A-Z]{3})\)/);
        if (fallbackMatch) {
          if (type === 'orig') currentOrig = fallbackMatch[1];
          if (type === 'dest') currentDest = fallbackMatch[1];
        }
      }, 250);
    }

    function initMap() {
      map = L.map('map', { zoomControl: false }).setView([34, -10], 4);
      L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png').addTo(map);
      dotsLayer = L.layerGroup().addTo(map);
      loadAirports();
    }

    async function fetchFlights() {

      const day = $('daySelect').value;
      $('flightSelect').innerHTML = '<option value="">Buscando vuelos...</option>';
      $('btnCompute').disabled = true;

      try {
        const res = await fetch(`/api/flights?orig=${currentOrig}&dest=${currentDest}&day=${day}`);
        const data = await res.json();

        const sel = $('flightSelect');
        sel.innerHTML = '';

        if (data.flights && data.flights.length > 0) {
          data.flights.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.depTime;
            opt.textContent = `${f.depTime} · Vuelo ${f.flight}`;
            sel.appendChild(opt);
          });
          $('flightBox').style.display = 'flex';
          $('manualTimeBox').style.display = 'none';
        } else {
          $('flightBox').style.display = 'none';
          $('manualTimeBox').style.display = 'flex';
        }
      } catch (e) {
        $('flightBox').style.display = 'none';
        $('manualTimeBox').style.display = 'flex';
      }
      $('btnCompute').disabled = false;
    }

    function getDummyCoords(iata) {
      let num = iata.charCodeAt(0) + iata.charCodeAt(1) + iata.charCodeAt(2);
      return { lat: (num % 80) - 20, lon: (num % 180) - 90 };
    }

    function addMinutesToTime(timeStr, minsToAdd) {
      let [h, m] = timeStr.split(':').map(Number);
      let date = new Date();
      date.setHours(h, m + minsToAdd, 0);
      return date.getHours().toString().padStart(2, '0') + ':' + date.getMinutes().toString().padStart(2, '0');
    }

    async function compute() {
      const btn = $('btnCompute');
      if (btn) btn.disabled = true;

      try {
        let startCoords = AIRPORTS_DB[currentOrig] || getDummyCoords(currentOrig);
        let endCoords = AIRPORTS_DB[currentDest] || getDummyCoords(currentDest);

        let baseTime = $('flightBox').style.display !== 'none' ? $('flightSelect').value : $('manualTime').value;
        if (!baseTime) baseTime = "12:00";

        map.setView([startCoords.lat, startCoords.lon], 4);
        dotsLayer.clearLayers();
        const dataPoints = [];
        const tableBody = document.querySelector('#dataTable tbody');
        tableBody.innerHTML = "";

        let modMinutes = 0, nubesMinutes = 0, catMinutes = 0, satShocks = 0;
        let maxWind = 0;
        let prevSat = null;

        for (let i = 0; i <= 12; i++) {
          const f = i / 12;
          const lat = startCoords.lat + (endCoords.lat - startCoords.lat) * f;
          const lon = startCoords.lon + (endCoords.lon - startCoords.lon) * f;

          const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=windspeed_250hPa,windspeed_300hPa,cloudcover_high,temperature_250hPa&forecast_days=1`);
          if (!r.ok) throw new Error("Error HTTP de Open-Meteo: " + r.status);
          const d = await r.json();

          if (!d.hourly || !d.hourly.windspeed_250hPa) {
            throw new Error("Datos meteorológicos no disponibles en esta ruta.");
          }

          const w250 = d.hourly.windspeed_250hPa[0];
          const w300 = d.hourly.windspeed_300hPa[0];
          const nubesAltas = d.hourly.cloudcover_high[0];
          const currentSat = d.hourly.temperature_250hPa[0];

          const vws = Math.abs(w250 - w300);
          const impactoNubes = nubesAltas > 50 ? (nubesAltas * 0.15) : 0;
          const atmosphericNoise = Math.sin(i * 0.8) * 3 + (Math.random() * 4);
          let edr = (vws * 0.45) + (w250 * 0.03) + impactoNubes + atmosphericNoise;
          if (edr < 2) edr = 2 + Math.random() * 2;

          let satWarning = "";
          if (prevSat !== null) {
            if (Math.abs(currentSat - prevSat) > 2.0) {
              satWarning = `<br><span class="badge b-sat">🧊 Choque Térmico</span>`;
              satShocks++;
            }
          }
          prevSat = currentSat;

          let causaTxt = "Estable"; let causaClase = "b-ok";
          if (edr > 15) {
            if (impactoNubes > 5) {
              causaTxt = "Nubes ☁️"; causaClase = "b-nubes"; nubesMinutes += 15;
            } else {
              causaTxt = "CAT 🌬️"; causaClase = "b-cat"; catMinutes += 15;
            }
          }

          if (w250 > maxWind) maxWind = w250;
          if (edr > 20) modMinutes += 15;

          const realTime = addMinutesToTime(baseTime, i * 15);
          const label = i === 0 ? `${currentOrig} (${realTime})` : i === 12 ? `${currentDest} (${realTime})` : realTime;

          dataPoints.push({ label, edr, vws, w250 });

          tableBody.innerHTML += `<tr>
          <td style="font-weight:bold;">${label}</td>
          <td style="color:#64748b;">${w250.toFixed(0)}</td>
          <td style="color:#ef4444; font-weight:600;">${vws.toFixed(0)}</td>
          <td style="line-height:1.4;"><span class="badge ${causaClase}">${causaTxt}</span>${satWarning}</td>
          <td style="color:${edr > 20 ? '#d97706' : '#059669'}; font-weight:bold;">${edr.toFixed(1)}</td>
        </tr>`;

          L.circleMarker([lat, lon], { radius: 6, color: edr > 20 ? '#fbbf24' : '#3b82f6', fillOpacity: 0.8 }).addTo(dotsLayer);
        }

        renderChart(dataPoints);
        generateBriefing(modMinutes, catMinutes, nubesMinutes, satShocks);
      } catch (err) {
        console.error("Analysis Error:", err);
        alert("Atención: Hubo un problema obteniendo los datos de la ruta. Por favor inténtalo de nuevo.");
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    function renderChart(points) {
      if (mainChart) mainChart.destroy();
      mainChart = new Chart($('mainChart'), {
        type: 'line',
        data: {
          labels: points.map(p => p.label),
          datasets: [
            { label: 'EDR', data: points.map(p => p.edr), borderColor: '#1e3a8a', borderWidth: 3, tension: 0.4, pointRadius: 2, fill: true, backgroundColor: 'rgba(59, 130, 246, 0.1)', yAxisID: 'y' },
            { label: 'Cizalladura (Δ)', data: points.map(p => p.vws), borderColor: '#ef4444', borderWidth: 2, borderDash: [4, 4], tension: 0.3, pointRadius: 0, fill: false, yAxisID: 'y1' }
          ]
        },
        options: {
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
          scales: {
            x: { ticks: { color: '#64748b', maxRotation: 45, minRotation: 45 } },
            y: { type: 'linear', display: true, position: 'left', min: 0, max: 60, ticks: { color: '#1e3a8a' } },
            y1: { type: 'linear', display: true, position: 'right', min: 0, max: 100, ticks: { color: '#ef4444' }, grid: { drawOnChartArea: false } }
          }
        }
      });
    }

    function generateBriefing(mod, cat, nubes, satShocks) {
      const briefing = $('briefing');
      const content = $('briefingContent');
      briefing.style.display = 'block';

      let text = `<p style="margin-top:0">Tu vuelo presenta <b>${mod} minutos</b> proyectados de turbulencia moderada o superior.</p>`;

      text += `<ul style="margin-bottom:0; padding-left: 20px;">`;
      if (mod > 0) {
        if (cat > 0) text += `<li><b>${cat} min de Aire Claro (CAT):</b> Turbulencia invisible causada por choques térmicos en el aire.</li>`;
        if (nubes > 0) text += `<li><b>${nubes} min Convectiva (Nubes):</b> Causada por el cruce de formaciones nubosas altas.</li>`;
      } else {
        text += `<li><b>Inestabilidad baja:</b> Vuelo en condiciones atmosféricas óptimas.</li>`;
      }

      if (satShocks > 0) {
        text += `<li><b style="color:#0891b2;">¡Atención SAT!</b> Se han detectado ${satShocks} cambios bruscos de temperatura. Posible cruce del núcleo del Jet Stream.</li>`;
      }
      text += `</ul>`;

      content.innerHTML = text;
    }

    // Lógica del Modal
    const modal = $('infoModal');
    $('btnOpenInfo').onclick = () => { modal.style.display = 'flex'; };
    $('btnCloseInfo').onclick = () => { modal.style.display = 'none'; };
    window.onclick = (e) => { if (e.target == modal) modal.style.display = "none"; }

    window.onload = initMap;
    $('btnCompute').onclick = compute;
  