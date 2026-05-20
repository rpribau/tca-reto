/* ═══════════════════════════════════════════════════════════
   Assist v10 — Dashboard JS (CMO Operational Decision Console)
   SPA Navigation, API integration, and Chart.js visualizations
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    // ─── Global State ──────────────────────────────────────────
    const state = {
        activePage: 'overview',
        charts: {},
        data: {
            summary: null,
            noshow: null,
            utilization: null,
            satisfaction: null,
            performance: null,
            trials: null,
            staffing: null,
            simulation: null
        }
    };

    // ─── DOM Elements ─────────────────────────────────────────
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    const pageTitle = document.getElementById('page-title');
    const btnRefresh = document.getElementById('btn-refresh');
    const lastUpdate = document.getElementById('last-update');
    const apiStatusBadge = document.getElementById('api-status');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');

    // ─── Color Palette for Charts ─────────────────────────────
    const colors = {
        accentBlue: '#58a6ff',
        accentGreen: '#3fb950',
        accentOrange: '#d29922',
        accentRed: '#f85149',
        accentPurple: '#bc8cff',
        textPrimary: '#e6edf3',
        textSecondary: '#8b949e',
        gridLine: 'rgba(255, 255, 255, 0.05)',
        tooltipBg: '#161b22',
        tooltipBorder: 'rgba(255, 255, 255, 0.1)'
    };

    // ─── Chart.js Defaults Configuration ──────────────────────
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = colors.textSecondary;
        Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
        Chart.defaults.plugins.legend.labels.color = colors.textPrimary;
        Chart.defaults.plugins.tooltip.backgroundColor = colors.tooltipBg;
        Chart.defaults.plugins.tooltip.borderColor = colors.tooltipBorder;
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.titleColor = colors.textPrimary;
        Chart.defaults.plugins.tooltip.bodyColor = colors.textPrimary;
    }

    // ─── Navigation ───────────────────────────────────────────
    function initNavigation() {
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const targetPage = item.getAttribute('data-page');
                setActivePage(targetPage);
                // Close sidebar on mobile
                sidebar.classList.remove('open');
            });
        });

        // Mobile Menu Toggle
        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }
    }

    function setActivePage(pageId) {
        state.activePage = pageId;

        // Toggle active nav item
        navItems.forEach(item => {
            if (item.getAttribute('data-page') === pageId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Toggle active page section
        pages.forEach(page => {
            if (page.id === `page-${pageId}`) {
                page.classList.add('active');
            } else {
                page.classList.remove('active');
            }
        });

        // Update Header Title
        const pageTitles = {
            overview: 'Consola de Staffing y Capacidad',
            noshow: 'Simulador Financiero & Agenda',
            model: 'Rendimiento de Modelos AI',
            experiments: 'Historial de Experimentos (Optuna)'
        };
        pageTitle.textContent = pageTitles[pageId] || 'Dashboard';

        // Re-render current page charts if we have data
        renderPageCharts(pageId);
    }

    // ─── API Integration & Fetching ───────────────────────────
    async function apiRequest(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`API error ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Request to ${endpoint} failed:`, error);
            updateApiStatus(false);
            return null;
        }
    }

    function updateApiStatus(isOnline) {
        if (apiStatusBadge) {
            const dot = apiStatusBadge.querySelector('.status-dot');
            const text = apiStatusBadge.querySelector('span:not(.status-dot)');
            if (isOnline) {
                dot.style.background = colors.accentGreen;
                text.textContent = 'API Conectada';
            } else {
                dot.style.background = colors.accentRed;
                text.textContent = 'Error de Conexión';
            }
        }
    }

    // ─── Business Simulator Controls & API Sync ───────────────
    let simTimeout;
    function initSimulatorBindings() {
        const costInput = document.getElementById('sim-consultation-cost');
        const overbookInput = document.getElementById('sim-overbooking-rate');
        const thresholdInput = document.getElementById('sim-threshold');
        const overtimeInput = document.getElementById('sim-overtime-cost');

        if (!costInput) return; // Not on page yet or HTML mismatch

        const triggerSimulationUpdate = () => {
            // Read input values
            const costVal = parseFloat(costInput.value);
            const overbookVal = parseFloat(overbookInput.value);
            const thresholdVal = parseFloat(thresholdInput.value);
            const overtimeVal = parseFloat(overtimeInput.value);

            // Update UI Labels immediately
            document.getElementById('val-sim-cost').textContent = `$${costVal}`;
            document.getElementById('val-sim-overbook').textContent = `${overbookVal}%`;
            document.getElementById('val-sim-threshold').textContent = `${thresholdVal}%`;
            document.getElementById('val-sim-overtime').textContent = `$${overtimeVal}`;

            // Debounced/Throttled POST request
            clearTimeout(simTimeout);
            simTimeout = setTimeout(async () => {
                try {
                    const response = await fetch('/v1/kpis/simulate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            threshold: thresholdVal / 100.0,
                            overbooking_rate: overbookVal,
                            consultation_cost: costVal,
                            hourly_overtime_cost: overtimeVal
                        })
                    });

                    if (!response.ok) throw new Error('Simulation failed');
                    const simResults = await response.json();
                    state.data.simulation = simResults;

                    // Update simulator view card elements
                    document.getElementById('sim-original-loss').textContent = `$${Math.round(simResults.financials.original_loss).toLocaleString()}`;
                    document.getElementById('sim-recovered-revenue').textContent = `$${Math.round(simResults.financials.recovered_revenue).toLocaleString()}`;
                    document.getElementById('sim-overtime-savings').textContent = `$${Math.round(simResults.financials.overtime_savings).toLocaleString()}`;
                    document.getElementById('sim-total-benefit').textContent = `$${Math.round(simResults.financials.total_annual_benefit).toLocaleString()}`;

                    document.getElementById('sim-util-original').textContent = `${simResults.utilization.original}%`;
                    document.getElementById('sim-util-optimized').textContent = `${simResults.utilization.optimized}%`;
                    document.getElementById('sim-nps-impact').textContent = `+${simResults.nps.nps_points_gain} pts`;
                    document.getElementById('sim-util-change').textContent = `+${simResults.utilization.change}%`;

                    // Update Card 4 (Ingreso Recuperado Anual) in overview page dynamically!
                    const elSatVal = document.getElementById('val-satisfaccion');
                    if (elSatVal) {
                        elSatVal.textContent = `$${Math.round(simResults.financials.total_annual_benefit).toLocaleString()}`;
                        const elSatTrend = document.getElementById('trend-satisfaccion');
                        if (elSatTrend) {
                            elSatTrend.className = 'kpi-trend up';
                            elSatTrend.textContent = `↑ ROI`;
                        }
                    }
                } catch (e) {
                    console.error('Error running simulation REST request:', e);
                }
            }, 150);
        };

        // Attach listeners
        [costInput, overbookInput, thresholdInput, overtimeInput].forEach(input => {
            input.addEventListener('input', triggerSimulationUpdate);
        });

        // Trigger once to bootstrap
        triggerSimulationUpdate();
    }

    async function loadAllData() {
        if (btnRefresh) btnRefresh.classList.add('spinning');
        lastUpdate.textContent = 'Actualizando...';

        updateApiStatus(true); // Assume online initially

        const [summary, noshow, utilization, satisfaction, performance, trials, staffing] = await Promise.all([
            apiRequest('/v1/kpis/summary'),
            apiRequest('/v1/kpis/noshow-rate'),
            apiRequest('/v1/kpis/utilization'),
            apiRequest('/v1/kpis/satisfaction'),
            apiRequest('/v1/kpis/model-performance'),
            apiRequest('/v1/kpis/optuna-trials'),
            apiRequest('/v1/kpis/staffing')
        ]);

        if (summary) state.data.summary = summary;
        if (noshow) state.data.noshow = noshow;
        if (utilization) state.data.utilization = utilization;
        if (satisfaction) state.data.satisfaction = satisfaction;
        if (performance) state.data.performance = performance;
        if (trials) state.data.trials = trials;
        if (staffing) state.data.staffing = staffing;

        // Check if any critical request failed
        if (!summary && !noshow) {
            updateApiStatus(false);
            lastUpdate.textContent = 'Error al actualizar';
        } else {
            updateApiStatus(true);
            const now = new Date();
            lastUpdate.textContent = `Actualizado: ${now.toLocaleTimeString()}`;
        }

        // Populate Static UI components (KPI cards, text, tables)
        populateKpiCards();
        populateStaffingAlerts();
        populateModelDetails();
        populateTrialsTable();

        // Initialize / trigger simulator bindings once we have data
        initSimulatorBindings();

        // Render current active page charts
        renderPageCharts(state.activePage);

        setTimeout(() => {
            if (btnRefresh) btnRefresh.classList.remove('spinning');
        }, 800);
    }

    // ─── UI Population functions ──────────────────────────────
    function populateKpiCards() {
        const data = state.data.summary;
        if (!data || !data.kpis) return;

        const kpis = data.kpis;

        // KPI 1: Ausentismo
        const elNoshowVal = document.getElementById('val-ausentismo');
        const elNoshowTrend = document.getElementById('trend-ausentismo');
        if (elNoshowVal && kpis.tasa_ausentismo) {
            elNoshowVal.textContent = `${kpis.tasa_ausentismo.value}${kpis.tasa_ausentismo.unit}`;
            setTrendBadge(elNoshowTrend, kpis.tasa_ausentismo.trend);
        }

        // KPI 2: Tiempo de Espera
        const elWaitVal = document.getElementById('val-espera');
        const elWaitTrend = document.getElementById('trend-espera');
        if (elWaitVal && kpis.tiempo_espera) {
            elWaitVal.textContent = `${kpis.tiempo_espera.value} ${kpis.tiempo_espera.unit}`;
            setTrendBadge(elWaitTrend, kpis.tiempo_espera.trend);
        }

        // KPI 3: Utilizacion
        const elUtilVal = document.getElementById('val-utilizacion');
        const elUtilTrend = document.getElementById('trend-utilizacion');
        if (elUtilVal && kpis.utilizacion_consultorios) {
            elUtilVal.textContent = `${kpis.utilizacion_consultorios.value}${kpis.utilizacion_consultorios.unit}`;
            setTrendBadge(elUtilTrend, kpis.utilizacion_consultorios.trend);
        }

        // KPI 4: Recalculated dynamically from Simulator (handled in initSimulatorBindings)
    }

    function setTrendBadge(badgeEl, trendValue) {
        if (!badgeEl) return;
        badgeEl.className = 'kpi-trend'; // reset
        
        let icon = '—';
        if (trendValue === 'up') {
            badgeEl.classList.add('up');
            icon = '↑';
        } else if (trendValue === 'down') {
            badgeEl.classList.add('down');
            icon = '↓';
        } else {
            badgeEl.classList.add('stable');
            icon = '→';
        }
        badgeEl.textContent = `${icon} ${trendValue.toUpperCase()}`;
    }

    function populateStaffingAlerts() {
        const staffing = state.data.staffing;
        const container = document.getElementById('staffing-alerts-list');
        if (!container) return;

        if (!staffing || !staffing.alerts || staffing.alerts.length === 0) {
            container.innerHTML = `
                <div class="alert-card" style="border-left-color: var(--accent-green); background: rgba(63, 185, 80, 0.05);">
                    <div class="alert-card-meta">
                        <span>ESTADO OPERATIVO CENTRAL</span>
                        <span>0 ALERTAS</span>
                    </div>
                    <div class="alert-card-text">Capacidad de personal óptima para todas las horas. No se proyectan cuellos de botella en Urgencias.</div>
                </div>
            `;
            return;
        }

        let html = '';
        staffing.alerts.forEach(alert => {
            const severityClass = alert.severity === 'critical' ? 'critical' : 'warning';
            const icon = alert.severity === 'critical' ? '🚨 CRÍTICO' : '⚠️ ALERTA';
            
            html += `
                <div class="alert-card ${severityClass}">
                    <div class="alert-card-meta">
                        <span>${icon} — Horario: ${alert.hour}</span>
                        <span>Pacientes Proyectados: ${alert.patients_arrival} | Médicos Activos: ${alert.current_doctors} (Requerido: ${alert.needed_doctors})</span>
                    </div>
                    <div class="alert-card-text">${alert.recommendation}</div>
                </div>
            `;
        });
        container.innerHTML = html;
    }

    function populateModelDetails() {
        const data = state.data.performance;
        if (!data || !data.his10) return;

        const cm = data.his10.confusion_matrix;
        
        // Confusion matrix values
        const tnEl = document.getElementById('cm-tn');
        const fpEl = document.getElementById('cm-fp');
        const fnEl = document.getElementById('cm-fn');
        const tpEl = document.getElementById('cm-tp');

        if (tnEl) tnEl.textContent = cm.tn.toLocaleString();
        if (fpEl) fpEl.textContent = cm.fp.toLocaleString();
        if (fnEl) fnEl.textContent = cm.fn.toLocaleString();
        if (tpEl) tpEl.textContent = cm.tp.toLocaleString();

        // Model meta info panel
        const infoEl = document.getElementById('model-info');
        if (infoEl) {
            const training = data.his10.training_info || {};
            
            let his05_status = '';
            if (data.his05) {
                his05_status = `
                    <div style="margin-top: 10px; border-top: 1px solid var(--border-color); padding-top: 10px;">
                        <span style="color: var(--accent-blue); font-weight:600;">Modelo HIS-05 (Saturación y Espera)</span><br>
                        • Algoritmo: <strong>${data.his05.model || 'XGBRegressor'}</strong><br>
                        • MAE en Validación: <strong>${data.his05.cv_MAE_mean || '—'}</strong> (± ${data.his05.cv_MAE_std || '—'})<br>
                        • R² en Validación: <strong>${data.his05.cv_R2_mean || '—'}</strong><br>
                        • Muestras de entrenamiento: <strong>${data.his05.n_samples ? data.his05.n_samples.toLocaleString() : '—'}</strong><br>
                        • Top features: <em>${data.his05.top10_features ? data.his05.top10_features.slice(0, 3).join(', ') : '—'}</em>
                    </div>
                `;
            } else {
                his05_status = `
                    <div style="margin-top: 10px; border-top: 1px solid var(--border-color); padding-top: 10px; color: var(--accent-orange);">
                        ⚠️ <strong>Modelo HIS-05 No Entrenado Aún:</strong><br>
                        El pipeline de forecasting de saturación y tiempo de espera no se ha ejecutado. Los valores en el dashboard de tiempos de espera se basan en promedios históricos de urgencias. Ejecute <code>kedro run --pipeline data_science_his05</code> en su terminal para entrenar el modelo.
                    </div>
                `;
            }

            infoEl.innerHTML = `
                <div>
                    <span style="color: var(--accent-blue); font-weight:600;">Modelo HIS-10 (No-Show Guard)</span><br>
                    • Algoritmo: <strong>LightGBM Classifier</strong><br>
                    • Muestras de validación: <strong>${training.test_size ? training.test_size.toLocaleString() : '—'}</strong><br>
                    • Features totales: <strong>${training.n_features || '—'}</strong><br>
                    • Umbral de decisión (Threshold): <strong>${training.threshold || 0.5}</strong><br>
                    • Fecha de entrenamiento: <strong>${data.his10.timestamp ? new Date(data.his10.timestamp).toLocaleString() : '—'}</strong>
                </div>
                ${his05_status}
            `;
        }
    }

    function populateTrialsTable() {
        const data = state.data.trials;
        const container = document.getElementById('trials-table-container');
        if (!container) return;

        if (!data || !data.trials || data.trials.length === 0) {
            container.innerHTML = `<p class="loading-text">No se encontraron registros de experimentos. Ejecute el pipeline de Optuna.</p>`;
            return;
        }

        // Sort trials by objective (ROC-AUC) descending
        const trials = [...data.trials].sort((a, b) => b.value - a.value);

        let html = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>N° Trial</th>
                        <th>Estado</th>
                        <th>ROC-AUC (Objetivo)</th>
                        <th>Learning Rate</th>
                        <th>Max Depth</th>
                        <th>Num Leaves</th>
                        <th>Subsample</th>
                        <th>Duración (s)</th>
                    </tr>
                </thead>
                <tbody>
        `;

        trials.forEach(trial => {
            const isBest = trial.state === 'COMPLETE' && Math.abs(trial.value - trials[0].value) < 1e-6;
            const rowClass = isBest ? 'class="best-row"' : '';
            const statusLabel = trial.state === 'COMPLETE' ? '✅' : '❌';
            
            html += `
                <tr ${rowClass}>
                    <td><strong>Trial #${trial.number}</strong> ${isBest ? '⭐ <small style="color: var(--accent-orange); font-weight:bold;">MEJOR</small>' : ''}</td>
                    <td>${statusLabel} ${trial.state}</td>
                    <td><strong style="color: var(--accent-green);">${trial.value.toFixed(5)}</strong></td>
                    <td>${trial.params_learning_rate ? parseFloat(trial.params_learning_rate).toFixed(5) : '—'}</td>
                    <td>${trial.params_max_depth || '—'}</td>
                    <td>${trial.params_num_leaves || '—'}</td>
                    <td>${trial.params_subsample ? parseFloat(trial.params_subsample).toFixed(3) : '—'}</td>
                    <td>${trial.duration ? (parseFloat(trial.duration).toFixed(1) + 's') : '—'}</td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
        `;
        container.innerHTML = html;
    }

    // ─── Chart Rendering Controllers ──────────────────────────
    function renderPageCharts(pageId) {
        // Destroy existing charts of the active page to allow redraw
        destroyChartsForPage(pageId);

        if (pageId === 'overview') {
            renderOverviewCharts();
        } else if (pageId === 'noshow') {
            renderNoshowCharts();
        } else if (pageId === 'model') {
            renderModelCharts();
        } else if (pageId === 'experiments') {
            renderExperimentsCharts();
        }
    }

    function destroyChartsForPage(pageId) {
        const pageChartIds = {
            overview: ['chart-noshow-area', 'chart-util-area', 'chart-satisfaction'],
            noshow: ['chart-noshow-month', 'chart-noshow-day', 'chart-noshow-specialty'],
            model: ['chart-model-metrics'],
            experiments: ['chart-optuna-auc']
        };

        const ids = pageChartIds[pageId] || [];
        ids.forEach(id => {
            if (state.charts[id]) {
                state.charts[id].destroy();
                delete state.charts[id];
            }
        });
    }

    // ─── Specific Charts Builders ─────────────────────────────
    
    function renderOverviewCharts() {
        const noshowData = state.data.noshow;
        const satData = state.data.satisfaction;
        const staffing = state.data.staffing;

        if (typeof Chart === 'undefined') return;

        // 1. Chart: Staffing e Indicación de Saturación (HIS-05) - Double Axis Line/Bar Chart
        if (staffing && staffing.schedule && document.getElementById('chart-util-area')) {
            const ctx = document.getElementById('chart-util-area').getContext('2d');
            const labels = staffing.schedule.map(s => s.hour_label);
            const patients = staffing.schedule.map(s => s.patients_arrival);
            const currentDocs = staffing.schedule.map(s => s.current_doctors);
            const neededDocs = staffing.schedule.map(s => s.needed_doctors);

            state.charts['chart-util-area'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Pacientes Proyectados (HIS-05)',
                            data: patients,
                            type: 'line',
                            borderColor: colors.accentBlue,
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.35,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Médicos Requeridos (IA)',
                            data: neededDocs,
                            type: 'bar',
                            backgroundColor: 'rgba(210, 153, 34, 0.45)',
                            borderColor: colors.accentOrange,
                            borderWidth: 1.5,
                            borderRadius: 4,
                            yAxisID: 'y1'
                        },
                        {
                            label: 'Médicos Actuales (Fijo)',
                            data: currentDocs,
                            type: 'line',
                            borderColor: colors.textSecondary,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            borderWidth: 2,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: colors.gridLine } },
                        y: { 
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Pacientes/Hora', color: colors.textSecondary },
                            grid: { color: colors.gridLine }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Médicos Activos', color: colors.textSecondary },
                            grid: { drawOnChartArea: false },
                            min: 0,
                            suggestedMax: 6,
                            ticks: { stepSize: 1 }
                        }
                    }
                }
            });
        }

        // 2. Chart: Ausentismo por Área
        if (noshowData && noshowData.by_area && document.getElementById('chart-noshow-area')) {
            const ctx = document.getElementById('chart-noshow-area').getContext('2d');
            const labels = Object.keys(noshowData.by_area);
            const values = Object.values(noshowData.by_area);

            state.charts['chart-noshow-area'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Tasa de Ausentismo (%)',
                        data: values,
                        backgroundColor: 'rgba(248, 81, 73, 0.65)',
                        borderColor: colors.accentRed,
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: colors.gridLine } },
                        y: { 
                            grid: { color: colors.gridLine },
                            ticks: { callback: value => `${value}%` }
                        }
                    }
                }
            });
        }

        // 3. Chart: Componentes de Satisfacción de Marca
        if (satData && satData.components && document.getElementById('chart-satisfaction')) {
            const ctx = document.getElementById('chart-satisfaction').getContext('2d');
            const comp = satData.components;

            state.charts['chart-satisfaction'] = new Chart(ctx, {
                type: 'polarArea',
                data: {
                    labels: ['Control Ausentismo', 'Flujo de Espera', 'Utilización Slots'],
                    datasets: [{
                        data: [comp.noshow_score, comp.wait_score, comp.utilization_score],
                        backgroundColor: [
                            'rgba(63, 185, 80, 0.65)',  // green
                            'rgba(210, 153, 34, 0.65)', // orange
                            'rgba(88, 166, 255, 0.65)'  // blue
                        ],
                        borderColor: '#161b22',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            grid: { color: colors.gridLine },
                            angleLines: { color: colors.gridLine },
                            ticks: { display: false },
                            max: 100
                        }
                    },
                    plugins: {
                        legend: { position: 'right' }
                    }
                }
            });
        }
    }

    function renderNoshowCharts() {
        const noshowData = state.data.noshow;
        if (!noshowData || typeof Chart === 'undefined') return;

        // 1. Chart: Tendencia Mensual del Ausentismo
        if (noshowData.by_month && document.getElementById('chart-noshow-month')) {
            const ctx = document.getElementById('chart-noshow-month').getContext('2d');
            
            const monthNames = {
                "1": "Ene", "2": "Feb", "3": "Mar", "4": "Abr", "5": "May", "6": "Jun",
                "7": "Jul", "8": "Ago", "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dic"
            };
            const rawLabels = Object.keys(noshowData.by_month);
            const labels = rawLabels.map(k => monthNames[k] || `Mes ${k}`);
            const values = Object.values(noshowData.by_month);

            state.charts['chart-noshow-month'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Tasa de Ausentismo (%)',
                        data: values,
                        borderColor: colors.accentOrange,
                        backgroundColor: 'rgba(210, 153, 34, 0.15)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: colors.accentOrange
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: colors.gridLine } },
                        y: { 
                            grid: { color: colors.gridLine },
                            ticks: { callback: value => `${value}%` }
                        }
                    }
                }
            });
        }

        // 2. Chart: Ausentismo por Día de la Semana
        if (noshowData.by_day_of_week && document.getElementById('chart-noshow-day')) {
            const ctx = document.getElementById('chart-noshow-day').getContext('2d');
            
            const dayNames = {
                "0": "Lunes", "1": "Martes", "2": "Miércoles", "3": "Jueves", "4": "Viernes", "5": "Sábado", "6": "Domingo"
            };
            const rawKeys = Object.keys(noshowData.by_day_of_week);
            const labels = rawKeys.map(k => dayNames[k] || `Día ${k}`);
            const values = Object.values(noshowData.by_day_of_week);

            state.charts['chart-noshow-day'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Tasa de Ausentismo (%)',
                        data: values,
                        backgroundColor: 'rgba(210, 153, 34, 0.65)',
                        borderColor: colors.accentOrange,
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: colors.gridLine } },
                        y: { 
                            grid: { color: colors.gridLine },
                            ticks: { callback: value => `${value}%` }
                        }
                    }
                }
            });
        }

        // 3. Chart: Top 10 Especialidades
        if (noshowData.by_specialty && document.getElementById('chart-noshow-specialty')) {
            const ctx = document.getElementById('chart-noshow-specialty').getContext('2d');
            const labels = Object.keys(noshowData.by_specialty);
            const values = Object.values(noshowData.by_specialty);

            state.charts['chart-noshow-specialty'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Tasa de Ausentismo (%)',
                        data: values,
                        backgroundColor: 'rgba(248, 81, 73, 0.65)',
                        borderColor: colors.accentRed,
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { 
                            grid: { color: colors.gridLine },
                            ticks: { callback: value => `${value}%` }
                        },
                        y: { grid: { color: 'transparent' } }
                    }
                }
            });
        }
    }

    function renderModelCharts() {
        const perfData = state.data.performance;
        if (!perfData || !perfData.his10 || !perfData.his10.metrics || typeof Chart === 'undefined') return;

        // 1. Chart: Model Metrics (Radar Chart)
        if (document.getElementById('chart-model-metrics')) {
            const ctx = document.getElementById('chart-model-metrics').getContext('2d');
            const metrics = perfData.his10.metrics;

            const labels = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy'];
            const datasetValues = [
                metrics.roc_auc,
                metrics.pr_auc,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.accuracy
            ];

            const datasets = [{
                label: 'HIS-10: No-Show Guard (Clasificación)',
                data: datasetValues,
                backgroundColor: 'rgba(88, 166, 255, 0.2)',
                borderColor: colors.accentBlue,
                borderWidth: 2,
                pointBackgroundColor: colors.accentBlue,
                pointBorderColor: '#fff'
            }];

            state.charts['chart-model-metrics'] = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            grid: { color: colors.gridLine },
                            angleLines: { color: colors.gridLine },
                            ticks: { color: colors.textSecondary, backdropColor: 'transparent', stepSize: 0.2 },
                            min: 0,
                            max: 1.0
                        }
                    }
                }
            });
        }
    }

    function renderExperimentsCharts() {
        const trialsData = state.data.trials;
        if (!trialsData || !trialsData.trials || trialsData.trials.length === 0 || typeof Chart === 'undefined') return;

        // 1. Chart: Optuna ROC-AUC history (Line Chart)
        if (document.getElementById('chart-optuna-auc')) {
            const ctx = document.getElementById('chart-optuna-auc').getContext('2d');
            
            // Sort trials chronologically by number
            const trials = [...trialsData.trials].sort((a, b) => a.number - b.number);
            const labels = trials.map(t => `Trial #${t.number}`);
            const values = trials.map(t => t.value);

            // Compute running best
            let currentBest = -Infinity;
            const runningBestValues = values.map(val => {
                if (val > currentBest) currentBest = val;
                return currentBest;
            });

            state.charts['chart-optuna-auc'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'ROC-AUC del Trial',
                            data: values,
                            borderColor: colors.accentBlue,
                            backgroundColor: 'rgba(88, 166, 255, 0.05)',
                            borderWidth: 1.5,
                            pointRadius: 3,
                            tension: 0.1
                        },
                        {
                            label: 'Mejor ROC-AUC Histórico',
                            data: runningBestValues,
                            borderColor: colors.accentPurple,
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            tension: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: colors.gridLine } },
                        y: { 
                            grid: { color: colors.gridLine },
                            ticks: { precision: 4 }
                        }
                    }
                }
            });
        }
    }

    // ─── Initializer ──────────────────────────────────────────
    initNavigation();
    loadAllData();

    // Event listener for Refresh Button
    if (btnRefresh) {
        btnRefresh.addEventListener('click', () => {
            loadAllData();
        });
    }
});
