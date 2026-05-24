import os
import json
import pandas as pd

# Source result directories
RESULT_DIRS = {
    'nGNN': 'results/benchmark_ngnn',
    'MC-Static': 'results/benchmark_ngnn_mc_static',
    'MC-Streaming': 'results/benchmark_ngnn_mc_streaming',
    'MC-Static+Legacy': 'results/benchmark_ngnn_mc_legacy_static',
    'MC-Streaming+Legacy': 'results/benchmark_ngnn_mc_legacy_streaming'
}

# Define legacy training times per chain
LEGACY_TRAINING_TIMES = {
    'bsc': 29392.10609961,
    'ethereum': 72781.390528257,
    'polygon': 8700.951005221
}

OUTPUT_DIR = 'docs/work_reports/26-benchmark_visualization_and_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_csv_file(file_path):
    return pd.read_csv(file_path)

def parse_results():
    records = []
    
    # 1. Parse nGNN (JSON files)
    ngnn_dir = RESULT_DIRS['nGNN']
    for filename in os.listdir(ngnn_dir):
        if filename.endswith('.json') and filename.startswith('benchmark_results_'):
            chain = filename.replace('benchmark_results_', '').replace('.json', '')
            data = load_json_file(os.path.join(ngnn_dir, filename))
            for item in data:
                if item.get('model_name') == 'Legacy-Augmentation':
                    continue
                records.append({
                    'chain': chain,
                    'category': 'nGNN',
                    'model_name': item.get('model_name'),
                    'roc_auc': float(item.get('roc_auc', 0)) if item.get('roc_auc') is not None and not pd.isna(item.get('roc_auc')) else 0.0,
                    'pr_auc': float(item.get('pr_auc', 0)) if item.get('pr_auc') is not None and not pd.isna(item.get('pr_auc')) else 0.0,
                    'best_f1': float(item.get('best_f1', 0)) if item.get('best_f1') is not None and not pd.isna(item.get('best_f1')) else 0.0,
                    'elapsed_sec': float(item.get('elapsed_sec', 0)) if item.get('elapsed_sec') is not None and not pd.isna(item.get('elapsed_sec')) else 0.0,
                    'peak_ram_mb': float(item.get('peak_ram_mb', 0)) if item.get('peak_ram_mb') is not None and not pd.isna(item.get('peak_ram_mb')) else 0.0,
                    'peak_gpu_mb': float(item.get('peak_gpu_mb', 0)) if item.get('peak_gpu_mb') is not None and not pd.isna(item.get('peak_gpu_mb')) else 0.0,
                    'has_legacy': False
                })

    # Helper function for parsing CSVs
    def parse_csv_dir(category, folder_path, file_prefix, has_legacy):
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv') and (filename.startswith(file_prefix) or 'results' in filename or 'benchmark' in filename):
                chain = filename.replace(file_prefix, '').replace('streaming_results_', '').replace('mc_benchmark_', '').replace('.csv', '')
                df = load_csv_file(os.path.join(folder_path, filename))
                for _, row in df.iterrows():
                    model_name = row['model_name']
                    if model_name == 'Legacy-Augmentation':
                        continue
                    records.append({
                        'chain': chain,
                        'category': category,
                        'model_name': model_name,
                        'roc_auc': float(row['roc_auc']) if not pd.isna(row['roc_auc']) else 0.0,
                        'pr_auc': float(row['pr_auc']) if not pd.isna(row['pr_auc']) else 0.0,
                        'best_f1': float(row['best_f1']) if not pd.isna(row['best_f1']) else 0.0,
                        'elapsed_sec': float(row['elapsed_sec']) if not pd.isna(row['elapsed_sec']) else 0.0,
                        'peak_ram_mb': float(row['peak_ram_mb']) if not pd.isna(row['peak_ram_mb']) else 0.0,
                        'peak_gpu_mb': float(row['peak_gpu_mb']) if not pd.isna(row['peak_gpu_mb']) else 0.0,
                        'has_legacy': has_legacy
                    })

    # 2. Parse MC-Static (CSVs)
    parse_csv_dir('MC-Static', RESULT_DIRS['MC-Static'], 'mc_benchmark_', False)
    
    # 3. Parse MC-Streaming (CSVs)
    parse_csv_dir('MC-Streaming', RESULT_DIRS['MC-Streaming'], 'streaming_results_', False)
    
    # 4. Parse MC-Static+Legacy (CSVs)
    parse_csv_dir('MC-Static+Legacy', RESULT_DIRS['MC-Static+Legacy'], 'mc_benchmark_', True)
    
    # 5. Parse MC-Streaming+Legacy (CSVs)
    parse_csv_dir('MC-Streaming+Legacy', RESULT_DIRS['MC-Streaming+Legacy'], 'streaming_results_', True)

    df_all = pd.DataFrame(records)
    
    # Impute Legacy Preprocessing Time
    df_all['legacy_time_sec'] = df_all.apply(
        lambda r: LEGACY_TRAINING_TIMES[r['chain']] if r['has_legacy'] else 0.0, axis=1
    )
    df_all['total_time_sec'] = df_all['elapsed_sec'] + df_all['legacy_time_sec']
    
    return df_all

def generate_html_dashboard(df, output_path):
    # Convert data to clean records list for JavaScript embedding
    records = df.to_dict(orient='records')
    json_data = json.dumps(records, indent=2)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DLG-GNN Benchmark Interactive Dashboard</title>
    <!-- Outfit & Inter Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: rgba(17, 24, 39, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --color-indigo: #6366f1;
            --color-emerald: #10b981;
            --color-blue: #3b82f6;
            --color-amber: #f59e0b;
            --color-pink: #ec4899;
            --color-purple: #a855f7;
            --font-outfit: 'Outfit', sans-serif;
            --font-inter: 'Inter', sans-serif;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-inter);
            min-height: 100vh;
            padding: 2rem;
            background-image: 
                radial-gradient(at 10% 20%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(236, 72, 153, 0.1) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
        }

        header {
            margin-bottom: 2.5rem;
            text-align: center;
        }

        h1 {
            font-family: var(--font-outfit);
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #d8b4fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 40px rgba(99, 102, 241, 0.2);
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }

        /* Glassmorphism Card style */
        .glass-card {
            background: var(--bg-surface);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            padding: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.25);
            box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
        }

        /* Layout Grid */
        .grid-kpi {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .kpi-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }

        .kpi-card.indigo::before { background: var(--color-indigo); }
        .kpi-card.emerald::before { background: var(--color-emerald); }
        .kpi-card.pink::before { background: var(--color-pink); }
        .kpi-card.amber::before { background: var(--color-amber); }

        .kpi-title {
            font-family: var(--font-outfit);
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        .kpi-value {
            font-family: var(--font-outfit);
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
            margin-bottom: 0.5rem;
        }

        .kpi-desc {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .kpi-desc span {
            color: #ffffff;
            font-weight: 600;
        }

        .grid-main {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        @media (min-width: 1024px) {
            .grid-main {
                grid-template-columns: 280px 1fr;
            }
        }

        /* Interactive Controls Card */
        .controls {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .control-title {
            font-family: var(--font-outfit);
            font-size: 0.95rem;
            font-weight: 600;
            color: #ffffff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.25rem;
            margin-bottom: 0.5rem;
        }

        .btn-tab {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0.6rem 1rem;
            text-align: left;
            font-weight: 500;
            transition: all 0.2s ease;
            width: 100%;
        }

        .btn-tab:hover {
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }

        .btn-tab.active {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
            border-color: var(--color-indigo);
            color: #ffffff;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.15);
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.9rem;
            padding: 0.25rem 0;
            transition: color 0.2s ease;
        }

        .checkbox-label:hover {
            color: #ffffff;
        }

        .checkbox-label input {
            accent-color: var(--color-indigo);
            width: 16px;
            height: 16px;
        }

        .search-box {
            width: 100%;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: #ffffff;
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        .search-box:focus {
            outline: none;
            border-color: var(--color-indigo);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
            background: rgba(255, 255, 255, 0.05);
        }

        /* Visualization Chart Card */
        .chart-section {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 480px;
        }

        .chart-header {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }

        @media (min-width: 768px) {
            .chart-header {
                flex-direction: row;
                align-items: center;
            }
        }

        .chart-metric-selector {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 0.3rem;
            border-radius: 10px;
        }

        .btn-metric {
            background: transparent;
            border: none;
            border-radius: 7px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.5rem 0.85rem;
            transition: all 0.2s ease;
        }

        .btn-metric:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.05);
        }

        .btn-metric.active {
            background: var(--color-indigo);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .chart-container {
            position: relative;
            flex-grow: 1;
            width: 100%;
            height: 380px;
        }

        /* Results Table Card */
        .table-section {
            overflow: hidden;
        }

        .table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }

        .table-header h2 {
            font-family: var(--font-outfit);
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
        }

        .table-wrapper {
            overflow-x: auto;
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }

        th {
            background-color: rgba(255, 255, 255, 0.03);
            color: #ffffff;
            font-weight: 600;
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s ease;
        }

        th:hover {
            background-color: rgba(255, 255, 255, 0.06);
        }

        th::after {
            content: ' ↕';
            font-size: 0.75rem;
            color: var(--text-secondary);
            opacity: 0.5;
        }

        th.asc::after { content: ' ▲'; opacity: 1; color: var(--color-indigo); }
        th.desc::after { content: ' ▼'; opacity: 1; color: var(--color-indigo); }

        td {
            padding: 0.9rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-secondary);
            font-family: var(--font-inter);
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background-color: rgba(255, 255, 255, 0.02);
            color: #ffffff;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .badge-chain {
            background: rgba(59, 130, 246, 0.1);
            color: var(--color-blue);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .badge-polygon {
            background: rgba(168, 85, 247, 0.1);
            color: var(--color-purple);
            border: 1px solid rgba(168, 85, 247, 0.2);
        }

        .badge-bsc {
            background: rgba(245, 158, 11, 0.1);
            color: var(--color-amber);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        .badge-category {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .badge-ngnn {
            background: rgba(77, 150, 255, 0.1);
            color: var(--color-blue);
            border: 1px solid rgba(77, 150, 255, 0.2);
        }

        .badge-mc-static {
            background: rgba(107, 203, 119, 0.1);
            color: var(--color-emerald);
            border: 1px solid rgba(107, 203, 119, 0.2);
        }

        .badge-mc-stream {
            background: rgba(255, 217, 61, 0.1);
            color: #d1b81d;
            border: 1px solid rgba(255, 217, 61, 0.2);
        }

        .text-bold {
            font-weight: 600;
            color: #ffffff;
        }

        .text-highlight {
            color: var(--color-emerald);
            font-weight: 600;
        }

        .footer-note {
            margin-top: 2rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .footer-note h3 {
            color: #ffffff;
            margin-bottom: 0.5rem;
            font-family: var(--font-outfit);
        }

        .footer-note ul {
            margin-left: 1.25rem;
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>

    <header>
        <h1>DLG-GNN Sabotaging & Fraud Detection Engine</h1>
        <div class="subtitle">Decoupled GNN & Monte Carlo Pipeline Multi-Chain Benchmark Results Dashboard</div>
    </header>

    <!-- KPI Section -->
    <div class="grid-kpi" id="kpiContainer">
        <!-- Filled dynamically -->
    </div>

    <!-- Main Visuals & Controls Grid -->
    <div class="grid-main">
        <!-- Controls Column -->
        <div class="glass-card controls">
            <div class="control-group">
                <div class="control-title">Blockchain Network</div>
                <button class="btn-tab active" onclick="setChainFilter('all')">All Networks</button>
                <button class="btn-tab" onclick="setChainFilter('polygon')" style="margin-top:0.4rem">Polygon (34.5:1 imbalance)</button>
                <button class="btn-tab" onclick="setChainFilter('bsc')" style="margin-top:0.4rem">BSC (5.7:1 imbalance)</button>
                <button class="btn-tab" onclick="setChainFilter('ethereum')" style="margin-top:0.4rem">Ethereum (1.4:1 imbalance)</button>
            </div>

            <div class="control-group">
                <div class="control-title">Pipeline Variant</div>
                <label class="checkbox-label">
                    <input type="checkbox" value="nGNN" checked onchange="toggleCategoryFilter()">
                    <span>nGNN (Decoupled Static)</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" value="MC-Static" checked onchange="toggleCategoryFilter()">
                    <span>MC-Static (Uncertainty Base)</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" value="MC-Streaming" checked onchange="toggleCategoryFilter()">
                    <span>MC-Streaming (Realtime)</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" value="MC-Static+Legacy" checked onchange="toggleCategoryFilter()">
                    <span>MC-Static + Legacy Aug</span>
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" value="MC-Streaming+Legacy" checked onchange="toggleCategoryFilter()">
                    <span>MC-Streaming + Legacy Aug</span>
                </label>
            </div>

            <div class="control-group">
                <div class="control-title">Search Models</div>
                <input type="text" class="search-box" id="modelSearch" placeholder="Search by model label..." onkeyup="filterTableAndCharts()">
            </div>
        </div>

        <!-- Chart Column -->
        <div class="glass-card chart-section">
            <div class="chart-header">
                <h2 style="font-family: var(--font-outfit); font-size:1.4rem; font-weight:700; color:#fff;" id="chartTitle">ROC-AUC Detection Comparison</h2>
                <div class="chart-metric-selector">
                    <button class="btn-metric active" onclick="setMetric('roc_auc')">ROC-AUC</button>
                    <button class="btn-metric" onclick="setMetric('pr_auc')">PR-AUC</button>
                    <button class="btn-metric" onclick="setMetric('best_f1')">Best F1</button>
                    <button class="btn-metric" onclick="setMetric('elapsed_sec')">Inference Time</button>
                    <button class="btn-metric" onclick="setMetric('total_time_sec')">Total Time (w/ Aug)</button>
                    <button class="btn-metric" onclick="setMetric('peak_ram_mb')">Peak RAM</button>
                    <button class="btn-metric" onclick="setMetric('peak_gpu_mb')">Peak GPU</button>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="benchmarkChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Table Section -->
    <div class="glass-card table-section">
        <div class="table-header">
            <h2>Detailed Benchmark Records Table</h2>
            <div class="kpi-desc" id="rowCount">Showing 0 of 0 records</div>
        </div>
        <div class="table-wrapper">
            <table id="benchmarkTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('chain')">Chain</th>
                        <th onclick="sortTable('category')">Category</th>
                        <th onclick="sortTable('model_label')">Model Variant</th>
                        <th onclick="sortTable('roc_auc')">ROC-AUC</th>
                        <th onclick="sortTable('pr_auc')">PR-AUC</th>
                        <th onclick="sortTable('best_f1')">Best F1</th>
                        <th onclick="sortTable('elapsed_sec')">Inf. (Sec)</th>
                        <th onclick="sortTable('total_time_sec')">Total Time (Sec)</th>
                        <th onclick="sortTable('peak_ram_mb')">Peak RAM (MB)</th>
                        <th onclick="sortTable('peak_gpu_mb')">Peak GPU (MB)</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Dynamic -->
                </tbody>
            </table>
        </div>
        
        <div class="footer-note">
            <h3>💡 Technical & Analysis Notes</h3>
            <ul>
                <li><strong>Legacy Preprocessing Imputation:</strong> The legacy augmented models require training five offline GNN detectors. Pre-computation costs are <strong>8,701s for Polygon</strong>, <strong>29,392s for BSC</strong>, and <strong>72,781s for Ethereum</strong>. The <em>Total Time</em> includes this overhead. While inference is extremely fast, this massive offline preprocessing is the primary bottleneck.</li>
                <li><strong>Network Imbalance:</strong> Polygon suffers from extreme class imbalance (34.5:1), making ROC-AUC highly sensitive. PR-AUC and F1 offer much cleaner signals.</li>
                <li><strong>Monte Carlo (MC) Dropout:</strong> MC Dropout allows quantification of epistemic uncertainty, which prevents false positives. When coupled with Legacy Feature Augmentation, the MC gain becomes consistently positive.</li>
                <li><strong>Streaming Advantage:</strong> On Ethereum, streaming variants outperform static ones in ROC-AUC, confirming that temporal dependency in transaction sequences is successfully captured.</li>
            </ul>
        </div>
    </div>

    <script>
        const rawData = {json_data};

        let currentChain = 'all';
        let currentMetric = 'roc_auc';
        let selectedCategories = ['nGNN', 'MC-Static', 'MC-Streaming', 'MC-Static+Legacy', 'MC-Streaming+Legacy'];
        let sortColumn = 'roc_auc';
        let sortDirection = 'desc';
        let chartInstance = null;

        function initDashboard() {
            updateKPIs();
            renderTable();
            renderChart();
        }

        function updateKPIs() {
            let topAUC = { roc_auc: 0, model_label: '', chain: '' };
            let topF1 = { best_f1: 0, model_label: '', chain: '' };
            let lowestRAM = { peak_ram_mb: Infinity, model_label: '', chain: '' };
            let fastestInf = { elapsed_sec: Infinity, model_label: '', chain: '' };

            rawData.forEach(r => {
                if (r.roc_auc > topAUC.roc_auc) {
                    topAUC.roc_auc = r.roc_auc;
                    topAUC.model_label = r.model_label;
                    topAUC.chain = r.chain;
                }
                if (r.best_f1 > topF1.best_f1) {
                    topF1.best_f1 = r.best_f1;
                    topF1.model_label = r.model_label;
                    topF1.chain = r.chain;
                }
                if (r.peak_ram_mb < lowestRAM.peak_ram_mb && r.peak_ram_mb > 0) {
                    lowestRAM.peak_ram_mb = r.peak_ram_mb;
                    lowestRAM.model_label = r.model_label;
                    lowestRAM.chain = r.chain;
                }
                if (r.elapsed_sec < fastestInf.elapsed_sec && r.elapsed_sec > 0) {
                    fastestInf.elapsed_sec = r.elapsed_sec;
                    fastestInf.model_label = r.model_label;
                    fastestInf.chain = r.chain;
                }
            });

            const kpiHTML = `
                <div class="glass-card kpi-card indigo">
                    <div class="kpi-title">Best Detection Accuracy</div>
                    <div class="kpi-value">${topAUC.roc_auc.toFixed(4)}</div>
                    <div class="kpi-desc">ROC-AUC achieved by <span>${topAUC.model_label}</span> on <span>${topAUC.chain.toUpperCase()}</span></div>
                </div>
                <div class="glass-card kpi-card emerald">
                    <div class="kpi-title">Best F1 Score</div>
                    <div class="kpi-value">${topF1.best_f1.toFixed(4)}</div>
                    <div class="kpi-desc">F1 Score achieved by <span>${topF1.model_label}</span> on <span>${topF1.chain.toUpperCase()}</span></div>
                </div>
                <div class="glass-card kpi-card pink">
                    <div class="kpi-title">Lowest Memory Footprint</div>
                    <div class="kpi-value">${lowestRAM.peak_ram_mb.toLocaleString(undefined, {maximumFractionDigits: 1})} MB</div>
                    <div class="kpi-desc">Peak RAM usage by <span>${lowestRAM.model_label}</span> on <span>${lowestRAM.chain.toUpperCase()}</span></div>
                </div>
                <div class="glass-card kpi-card amber">
                    <div class="kpi-title">Fastest Inference Speed</div>
                    <div class="kpi-value">${fastestInf.elapsed_sec.toFixed(3)} s</div>
                    <div class="kpi-desc">Inference time by <span>${fastestInf.model_label}</span> on <span>${fastestInf.chain.toUpperCase()}</span></div>
                </div>
            `;
            document.getElementById('kpiContainer').innerHTML = kpiHTML;
        }

        function getFilteredData() {
            const searchVal = document.getElementById('modelSearch').value.toLowerCase();
            return rawData.filter(r => {
                const chainMatch = currentChain === 'all' || r.chain === currentChain;
                const catMatch = selectedCategories.includes(r.category);
                const searchMatch = r.model_label.toLowerCase().includes(searchVal);
                return chainMatch && catMatch && searchMatch;
            });
        }

        function sortData(data) {
            return data.sort((a, b) => {
                let valA = a[sortColumn];
                let valB = b[sortColumn];
                
                if (typeof valA === 'string') {
                    valA = valA.toLowerCase();
                    valB = valB.toLowerCase();
                }
                
                if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
                if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
                return 0;
            });
        }

        function renderTable() {
            const tbody = document.getElementById('tableBody');
            const filtered = getFilteredData();
            const sorted = sortData(filtered);
            
            document.getElementById('rowCount').textContent = `Showing ${sorted.length} of ${rawData.length} records`;

            let html = '';
            sorted.forEach(r => {
                let chainClass = r.chain === 'ethereum' ? 'badge-chain' : (r.chain === 'polygon' ? 'badge-polygon' : 'badge-bsc');
                let catClass = r.category.includes('Legacy') ? 'badge-mc-stream' : (r.category.includes('MC') ? 'badge-mc-static' : 'badge-ngnn');
                
                html += `
                    <tr>
                        <td><span class="badge ${chainClass}">${r.chain.toUpperCase()}</span></td>
                        <td><span class="badge ${catClass}">${r.category}</span></td>
                        <td class="text-bold">${r.model_label}</td>
                        <td class="${r.roc_auc > 0.95 ? 'text-highlight' : ''}">${r.roc_auc.toFixed(4)}</td>
                        <td>${r.pr_auc.toFixed(4)}</td>
                        <td>${r.best_f1.toFixed(4)}</td>
                        <td>${r.elapsed_sec.toFixed(3)}s</td>
                        <td>${r.total_time_sec.toLocaleString(undefined, {maximumFractionDigits: 1})}s</td>
                        <td>${r.peak_ram_mb.toLocaleString(undefined, {maximumFractionDigits: 1})} MB</td>
                        <td>${r.peak_gpu_mb.toLocaleString(undefined, {maximumFractionDigits: 1})} MB</td>
                    </tr>
                `;
            });
            
            tbody.innerHTML = html;
            updateTableHeaderClasses();
        }

        function updateTableHeaderClasses() {
            const thElements = document.querySelectorAll('th');
            const colMap = ['chain', 'category', 'model_label', 'roc_auc', 'pr_auc', 'best_f1', 'elapsed_sec', 'total_time_sec', 'peak_ram_mb', 'peak_gpu_mb'];
            
            thElements.forEach((th, idx) => {
                th.className = '';
                const colName = colMap[idx];
                if (colName === sortColumn) {
                    th.classList.add(sortDirection);
                }
            });
        }

        function sortTable(column) {
            if (sortColumn === column) {
                sortDirection = sortDirection === 'desc' ? 'asc' : 'desc';
            } else {
                sortColumn = column;
                sortDirection = (column.includes('auc') || column.includes('f1')) ? 'desc' : 'asc';
            }
            renderTable();
        }

        function setChainFilter(chain) {
            currentChain = chain;
            const buttons = document.querySelectorAll('.control-group:first-child .btn-tab');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            const idx = chain === 'all' ? 0 : (chain === 'polygon' ? 1 : (chain === 'bsc' ? 2 : 3));
            buttons[idx].classList.add('active');
            
            renderTable();
            renderChart();
        }

        function toggleCategoryFilter() {
            const checkboxes = document.querySelectorAll('.checkbox-label input');
            selectedCategories = [];
            checkboxes.forEach(cb => {
                if (cb.checked) selectedCategories.push(cb.value);
            });
            renderTable();
            renderChart();
        }

        function setMetric(metric) {
            currentMetric = metric;
            const buttons = document.querySelectorAll('.chart-metric-selector .btn-metric');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            const metricMap = ['roc_auc', 'pr_auc', 'best_f1', 'elapsed_sec', 'total_time_sec', 'peak_ram_mb', 'peak_gpu_mb'];
            const idx = metricMap.indexOf(metric);
            buttons[idx].classList.add('active');

            const titles = {
                'roc_auc': 'ROC-AUC Detection Accuracy Comparison',
                'pr_auc': 'PR-AUC (Precision-Recall AUC) Comparison',
                'best_f1': 'Best F1-Score Performance Comparison',
                'elapsed_sec': 'Model Inference/Fine-Tuning Execution Time',
                'total_time_sec': 'Total Execution Time (including Legacy GNN preprocessing)',
                'peak_ram_mb': 'Peak System RAM Allocation Footprint',
                'peak_gpu_mb': 'Peak GPU VRAM Footprint'
            };
            document.getElementById('chartTitle').textContent = titles[metric];

            renderChart();
        }

        function filterTableAndCharts() {
            renderTable();
            renderChart();
        }

        function renderChart() {
            const filtered = getFilteredData();
            let chartLabels = [];
            let datasets = [];
            
            const catColors = {
                'nGNN': '#4D96FF',
                'MC-Static': '#6BCB77',
                'MC-Streaming': '#FFD93D',
                'MC-Static+Legacy': '#FF6B6B',
                'MC-Streaming+Legacy': '#B983FF'
            };

            if (currentChain === 'all') {
                chartLabels = ['polygon', 'bsc', 'ethereum'];
                
                selectedCategories.forEach(cat => {
                    const dataPoints = chartLabels.map(chain => {
                        const items = filtered.filter(r => r.chain === chain && r.category === cat);
                        if (items.length === 0) return null;
                        return Math.max(...items.map(item => item[currentMetric]));
                    });
                    
                    datasets.push({
                        label: cat,
                        data: dataPoints,
                        backgroundColor: catColors[cat],
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        borderRadius: 6
                    });
                });
                
                chartLabels = chartLabels.map(l => l.toUpperCase());
            } else {
                const sortedModels = filtered.sort((a, b) => b[currentMetric] - a[currentMetric]);
                chartLabels = sortedModels.map(r => r.model_label);
                const dataPoints = sortedModels.map(r => r[currentMetric]);
                const backgroundColors = sortedModels.map(r => catColors[r.category]);
                
                datasets.push({
                    label: currentMetric.replace('_', ' ').toUpperCase(),
                    data: dataPoints,
                    backgroundColor: backgroundColors,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    borderRadius: 6
                });
            }

            if (chartInstance) {
                chartInstance.destroy();
            }

            const ctx = document.getElementById('benchmarkChart').getContext('2d');
            const isTimeOrMemory = ['elapsed_sec', 'total_time_sec', 'peak_ram_mb', 'peak_gpu_mb'].includes(currentMetric);
            const scaleType = isTimeOrMemory ? 'logarithmic' : 'linear';

            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: chartLabels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: scaleType,
                            beginAtZero: !isTimeOrMemory,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawTicks: false
                            },
                            ticks: {
                                color: '#9ca3af',
                                font: {
                                    family: 'Inter'
                                },
                                callback: function(value, index, ticks) {
                                    if (isTimeOrMemory) {
                                        return value.toLocaleString() + (currentMetric.includes('mb') ? ' MB' : ' s');
                                    }
                                    return value.toFixed(2);
                                }
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                color: '#9ca3af',
                                font: {
                                    family: 'Inter',
                                    size: 10
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: currentChain === 'all',
                            position: 'top',
                            labels: {
                                color: '#f3f4f6',
                                font: {
                                    family: 'Outfit',
                                    weight: 'bold'
                                }
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(17, 24, 39, 0.95)',
                            titleFont: {
                                family: 'Outfit',
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                family: 'Inter',
                                size: 12
                            },
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) label += ': ';
                                    let val = context.parsed.y;
                                    if (currentMetric.includes('sec') || currentMetric.includes('time')) {
                                        return label + val.toFixed(3) + ' seconds';
                                    } else if (currentMetric.includes('mb')) {
                                        return label + val.toLocaleString(undefined, {maximumFractionDigits:1}) + ' MB';
                                    }
                                    return label + val.toFixed(5);
                                }
                            }
                        }
                    }
                }
            });
        }

        window.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
"""
    
    html_content = html_content.replace('{json_data}', json_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Interactive HTML dashboard successfully generated at: {output_path}")

def main():
    print("Loading and parsing benchmark files...")
    df = parse_results()
    
    model_labels = {
        'Revision-L1-NGNN': 'nGNN L1',
        'Revision-L1+L2-NGNN': 'nGNN L1+L2',
        'Revision-Full-NGNN': 'nGNN Full',
        'L1-Base': 'L1-Base (Static)',
        'L1-MC': 'L1-MC (Static)',
        'L1+L2-Base': 'L1+L2-Base (Static)',
        'L1+L2-MC': 'L1+L2-MC (Static)',
        'L1-StreamMC': 'L1-MC (Stream)',
        'L1+L2-StreamMC': 'L1+L2-MC (Stream)',
        'L1-Base-Aug': 'L1-Base-Aug (Static)',
        'L1-MC-Aug': 'L1-MC-Aug (Static)',
        'L1+L2-Base-Aug': 'L1+L2-Base-Aug (Static)',
        'L1+L2-MC-Aug': 'L1+L2-MC-Aug (Static)',
        'L1-StreamMC-Aug': 'L1-MC-Aug (Stream)',
        'L1+L2-StreamMC-Aug': 'L1+L2-MC-Aug (Stream)'
    }
    df['model_label'] = df['model_name'].map(model_labels).fillna(df['model_name'])
    
    # Export clean combined dataset to csv for reference
    df.to_csv(os.path.join(OUTPUT_DIR, 'combined_results.csv'), index=False)
    print(f"Combined clean results saved to {OUTPUT_DIR}/combined_results.csv")
    
    # Generate HTML Dashboard
    generate_html_dashboard(df, os.path.join(OUTPUT_DIR, 'benchmark_dashboard.html'))
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set theme and plot standard images
        sns.set_theme(style="whitegrid")
        
        category_colors = {
            'nGNN': '#4D96FF',
            'MC-Static': '#6BCB77',
            'MC-Streaming': '#FFD93D',
            'MC-Static+Legacy': '#FF6B6B',
            'MC-Streaming+Legacy': '#B983FF'
        }
        
        # Plot 1: ROC-AUC Comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df, 
            x='chain', 
            y='roc_auc', 
            hue='category', 
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5
        )
        plt.title('ROC-AUC Comparison Across Blockchain Networks & Pipeline Variants')
        plt.ylabel('ROC-AUC')
        plt.xlabel('Blockchain Network')
        plt.ylim(0.5, 1.02)
        plt.legend(title='Pipeline Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_auc_comparison.png'), bbox_inches='tight')
        plt.close()
        
        # Plot 2: PR-AUC Comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df, 
            x='chain', 
            y='pr_auc', 
            hue='category', 
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5
        )
        plt.title('PR-AUC (Precision-Recall AUC) Comparison Across Networks')
        plt.ylabel('PR-AUC')
        plt.xlabel('Blockchain Network')
        plt.ylim(0.8, 1.01)
        plt.legend(title='Pipeline Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pr_auc_comparison.png'), bbox_inches='tight')
        plt.close()
        
        # Plot 3: Best F1 Score
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df, 
            x='chain', 
            y='best_f1', 
            hue='category', 
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5
        )
        plt.title('Best F1 Score Comparison Across Networks')
        plt.ylabel('Best F1 Score')
        plt.xlabel('Blockchain Network')
        plt.ylim(0.7, 1.01)
        plt.legend(title='Pipeline Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'f1_comparison.png'), bbox_inches='tight')
        plt.close()

        # Plot 4: Execution Time Comparison (Log scale for elapsed time)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left subplot: Inference/fine-tuning time alone (elapsed_sec)
        sns.barplot(
            data=df,
            x='chain',
            y='elapsed_sec',
            hue='category',
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5,
            ax=ax1
        )
        ax1.set_title('Model Execution / Inference Time Alone (log scale)')
        ax1.set_ylabel('Time (Seconds) - Log Scale')
        ax1.set_yscale('log')
        ax1.set_xlabel('Blockchain Network')
        ax1.get_legend().remove()
        
        # Right subplot: Total Time including Legacy Training
        sns.barplot(
            data=df,
            x='chain',
            y='total_time_sec',
            hue='category',
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5,
            ax=ax2
        )
        ax2.set_title('Total Time (Inference + Legacy Training/Preprocessing, log scale)')
        ax2.set_ylabel('Total Time (Seconds) - Log Scale')
        ax2.set_yscale('log')
        ax2.set_xlabel('Blockchain Network')
        ax2.legend(title='Pipeline Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.suptitle('Temporal Cost Comparison (Inference vs. Full Pipeline Overhead)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'elapsed_time_comparison.png'), bbox_inches='tight')
        plt.close()

        # Plot 5: Peak RAM & GPU Usage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # RAM Usage
        sns.barplot(
            data=df,
            x='chain',
            y='peak_ram_mb',
            hue='category',
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5,
            ax=ax1
        )
        ax1.set_title('Peak RAM Usage (MB) Across Networks')
        ax1.set_ylabel('Peak RAM (MB)')
        ax1.set_xlabel('Blockchain Network')
        ax1.get_legend().remove()
        
        # GPU Usage
        sns.barplot(
            data=df,
            x='chain',
            y='peak_gpu_mb',
            hue='category',
            palette=category_colors,
            edgecolor='black',
            linewidth=0.5,
            ax=ax2
        )
        ax2.set_title('Peak GPU VRAM Usage (MB) Across Networks')
        ax2.set_ylabel('Peak GPU VRAM (MB)')
        ax2.set_xlabel('Blockchain Network')
        ax2.legend(title='Pipeline Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.suptitle('Computational Resource Footprint Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'resource_usage_comparison.png'), bbox_inches='tight')
        plt.close()
        
        print("All static comparison plots generated successfully in: " + OUTPUT_DIR)
    
    except ImportError:
        print("Warning: matplotlib or seaborn is not available yet. Only HTML dashboard has been created.")
        print("Please wait for package installation to complete and re-run this script to generate PNG images.")

    # High-level summary printed to console
    print("\n" + "="*80)
    print("BENCHMARK PERFORMANCE SUMMARY (Best per Category per Chain)")
    print("="*80)
    for chain in ['polygon', 'bsc', 'ethereum']:
        print(f"\n--- Blockchain: {chain.upper()} ---")
        chain_df = df[df['chain'] == chain]
        best_rows = chain_df.loc[chain_df.groupby('category')['roc_auc'].idxmax()]
        for _, row in best_rows.sort_values(by='roc_auc', ascending=False).iterrows():
            print(f"[{row['category']}] {row['model_name']}: "
                  f"ROC-AUC={row['roc_auc']:.4f} | PR-AUC={row['pr_auc']:.4f} | F1={row['best_f1']:.4f} | "
                  f"Execution Time={row['elapsed_sec']:.2f}s | Total Time (with Legacy)={row['total_time_sec']:.2f}s | "
                  f"Peak RAM={row['peak_ram_mb']:.1f} MB | Peak GPU={row['peak_gpu_mb']:.1f} MB")
    print("="*80)

if __name__ == '__main__':
    main()
