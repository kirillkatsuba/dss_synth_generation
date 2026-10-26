#!/usr/bin/env python3
"""Inspect ground-truth IOPS/latency clouds and export their original rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dss_clouds_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dss_clouds_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

PALETTE = [
    "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777",
    "#65a30d", "#4f46e5", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#393b79", "#637939",
    "#8c6d31", "#843c39",
]


def prepare_payload(
    frame: pd.DataFrame, source: Path, limits: dict[str, list[float]],
    iops_divisor: float, latency_divisor: float,
) -> dict[str, Any]:
    required = {"iops", "lat"}
    if not required.issubset(frame.columns):
        raise ValueError("Input CSV must contain iops and lat.")
    if {"source_row", "cloud_id", "cloud_name"} & set(frame.columns):
        raise ValueError("Input contains reserved annotation column names.")
    numerical = frame[["iops", "lat"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numerical.to_numpy(dtype=float)).all():
        raise ValueError("Ground-truth IOPS/latency must be finite; no rows are silently dropped.")
    context = frame.drop(columns=["iops", "lat"])
    if len(context.columns):
        codes = context.groupby(list(context.columns), sort=False, dropna=False).ngroup().astype(int)
        configurations = context.assign(_config=codes).drop_duplicates("_config").sort_values("_config").drop(columns="_config")
        configs = json.loads(configurations.to_json(orient="records"))
    else:
        codes = pd.Series(np.zeros(len(frame), dtype=int))
        configs = [{}]
    rows = [list(row) for row in zip(numerical.iops.tolist(), numerical.lat.tolist(), codes.tolist())]
    x = numerical.iops.to_numpy() / iops_divisor
    y = numerical.lat.to_numpy() / latency_divisor
    visible = (x >= limits["iops"][0]) & (x <= limits["iops"][1]) & (y >= limits["lat"][0]) & (y <= limits["lat"][1])
    return {
        "source": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "columns": frame.columns.tolist(), "rows": rows, "configurations": configs,
        "limits": limits, "iops_divisor": iops_divisor, "latency_divisor": latency_divisor,
        "visible_rows": int(visible.sum()), "palette": PALETTE,
    }


def load_annotations(path: Path, payload: dict[str, Any]) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    annotations = json.loads(path.read_text())
    if annotations.get("source_sha256") != payload["sha256"]:
        raise ValueError("Annotations were created for a different source CSV.")
    if annotations.get("rows") != len(payload["rows"]):
        raise ValueError("Annotation row count does not match the source CSV.")
    labels = np.zeros(len(payload["rows"]), dtype=int)
    groups: dict[int, dict[str, Any]] = {}
    for group in annotations["clouds"]:
        identifier = group["id"]
        if type(identifier) is not int or identifier <= 0 or identifier in groups:
            raise ValueError("Cloud IDs must be positive, unique integers.")
        if any(type(index) is not int for index in group["source_rows"]):
            raise ValueError("Cloud source rows must be integer zero-based indices.")
        indices = np.asarray(group["source_rows"], dtype=int)
        if (indices < 0).any() or (indices >= len(labels)).any() or len(np.unique(indices)) != len(indices):
            raise ValueError("Cloud source rows must be unique valid zero-based indices.")
        if (labels[indices] != 0).any():
            raise ValueError("Each source row can belong to at most one cloud.")
        labels[indices] = identifier
        groups[identifier] = group
    return labels, groups


def save_plot(frame: pd.DataFrame, payload: dict[str, Any], output: Path, labels: np.ndarray | None = None, groups: dict[int, dict[str, Any]] | None = None) -> None:
    x = frame.iops / payload["iops_divisor"]
    y = frame.lat / payload["latency_divisor"]
    many_groups = groups is not None and len(groups) > 8
    fig, ax = plt.subplots(figsize=(10, 7) if many_groups else (7, 5.5))
    if labels is None:
        ax.scatter(x, y, s=0.6, alpha=0.4, color="#2563eb", linewidths=0)
    else:
        ax.scatter(x[labels == 0], y[labels == 0], s=0.6, alpha=0.18, color="#9ca3af", linewidths=0)
        for identifier, group in groups.items():
            selected = labels == identifier
            ax.scatter(x[selected], y[selected], s=1.5, alpha=0.6, color=group["color"], linewidths=0, label=f"{group['name']} (n={selected.sum()})")
        if groups:
            if many_groups:
                ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=4)
            else:
                ax.legend(fontsize=8, loc="best", markerscale=4)
    ax.set(xlim=payload["limits"]["iops"], ylim=payload["limits"]["lat"],
           xlabel=f"IOPS / {payload['iops_divisor']:g}", ylabel=f"Latency / {payload['latency_divisor']:g}",
           title="Ground truth: latency vs. IOPS")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_cloud_panels(frame, payload, output, labels, groups):
    """Show local views without changing cloud membership or source row indices."""
    if not groups:
        return
    columns = min(4, len(groups))
    rows = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.8, rows * 3.1), squeeze=False)
    x = frame.iops / payload["iops_divisor"]
    y = frame.lat / payload["latency_divisor"]
    for ax, (identifier, group) in zip(axes.flat, groups.items()):
        selected = labels == identifier
        if not selected.any():
            ax.set_visible(False)
            continue
        xs, ys = x[selected], y[selected]
        xpad = max(float(xs.max() - xs.min()) * 0.15, 0.15)
        ypad = max(float(ys.max() - ys.min()) * 0.15, 0.5)
        ax.scatter(x[~selected], y[~selected], s=2, color="#9ca3af", alpha=0.15, linewidths=0)
        ax.scatter(xs, ys, s=5, color=group["color"], alpha=0.7, linewidths=0)
        ax.set(xlim=(max(0, xs.min()-xpad), xs.max()+xpad),
               ylim=(max(0, ys.min()-ypad), ys.max()+ypad),
               xlabel=f"IOPS / {payload['iops_divisor']:g}",
               ylabel=f"Latency / {payload['latency_divisor']:g}")
        experiment_ids = frame.loc[selected, "id"].unique() if "id" in frame else []
        subtitle = str(experiment_ids[0]) if len(experiment_ids) == 1 else f"IDs: {len(experiment_ids)}"
        ax.set_title(f"{group['name']} (n={selected.sum()})\n{subtitle}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle("Ground truth clouds: local views (other experiments in gray)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output, dpi=200)
    plt.close(fig)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ground truth: IOPS / latency clouds</title>
<style>
.details>section{min-width:0}.toolbar label{max-width:100%;min-width:0}input[type=file]{width:235px;min-width:0}
*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;color:#20252a;background:#fff}main{max-width:1320px;margin:auto;padding:20px}h1{font-size:21px;font-weight:600;margin:0 0 12px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px}button,input,select{font:inherit;padding:7px 10px;border:1px solid #c7cdd3;border-radius:4px;background:#fff;color:inherit}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button.primary{background:#20252a;color:white;border-color:#20252a}label{display:flex;gap:6px;align-items:center}.status{color:#54616a;font-variant-numeric:tabular-nums;line-height:1.6}#plot{height:590px;width:100%}.details{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 9px;border-bottom:1px solid #e2e5e9;text-align:left;white-space:nowrap}th{font-weight:600}h2{font-size:15px;font-weight:600;margin:10px 0}.swatch{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:50%}#selection-count{font-weight:600}#cloud-name{width:150px}#error{color:#b91c1c;margin:8px 0} @media(max-width:760px){main{padding:12px}.details{grid-template-columns:1fr;gap:8px}#plot{height:480px}.toolbar{gap:8px}label{flex-wrap:wrap}} @media(max-width:390px){button,input,select{max-width:100%}#cloud-name{width:125px}}
</style></head><body><main>
<h1>Ground truth: IOPS / latency clouds</h1>
<div class="toolbar">
 <label>Cloud <select id="cloud"></select></label><input id="cloud-name" aria-label="Cloud name" value="Cloud 1">
 <button id="new-cloud" title="Create a cloud">New cloud</button>
 <button id="assign" class="primary" disabled>Assign selection</button>
 <button id="unassign" disabled>Unassign selection</button>
 <label>Color <select id="color-by"><option value="cloud">Cloud</option></select></label>
</div>
<div class="toolbar">
 <button id="selected-csv" disabled>Selection CSV</button><button id="clouds-csv" disabled>Clouds CSV</button>
 <button id="export-json">Annotations JSON</button><label>Import <input id="import-json" type="file" accept=".json,application/json"></label>
</div>
<div class="status" aria-live="polite"><span id="overview"></span> &middot; Selected: <span id="selection-count">0</span> &middot; Assigned: <span id="assigned-count">0</span></div>
<div id="error" role="alert"></div>
<div id="plot" role="img" aria-label="Ground truth IOPS and latency scatter, with lasso selection"></div>
<div class="details">
 <section><h2>Clouds</h2><div class="scroll"><table><thead><tr><th>Cloud</th><th>Rows</th><th>IDs</th><th>Median IOPS</th><th>Median latency</th></tr></thead><tbody id="cloud-table"></tbody></table></div></section>
 <section><h2>Selected configurations</h2><div class="scroll"><table><thead><tr><th>ID</th><th>Rows</th><th>Block size</th><th>Jobs</th><th>Depth</th><th>IO</th><th>RAID</th><th>Disks</th></tr></thead><tbody id="config-table"></tbody></table></div></section>
</div>
</main><script>__PLOTLY__</script><script type="application/json" id="source-data">__PAYLOAD__</script>
<script>
const source=JSON.parse(document.getElementById('source-data').textContent);
const plot=document.getElementById('plot');
const rows=source.rows, configs=source.configurations;
const x=rows.map(r=>r[0]/source.iops_divisor), y=rows.map(r=>r[1]/source.latency_divisor);
let groups=[{id:1,name:'Cloud 1',color:source.palette[0]}], labels=new Array(rows.length).fill(0), selected=[], geometry=null;
if(source.initial_annotations){groups=source.initial_annotations.clouds.map(g=>({id:g.id,name:g.name,color:g.color}));for(const group of source.initial_annotations.clouds)for(const index of group.source_rows)labels[index]=group.id;if(!groups.length)groups=[{id:1,name:'Cloud 1',color:source.palette[0]}]}
const byId=id=>document.getElementById(id);
const escapeText=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const median=values=>{const sorted=values.slice().sort((a,b)=>a-b),n=sorted.length;return n?(n%2?sorted[(n-1)/2]:(sorted[n/2-1]+sorted[n/2])/2):null};
const format=v=>v===null?'':v.toLocaleString(undefined,{maximumFractionDigits:3});
const hover=rows.map((row,i)=>{const c=configs[row[2]];return `Row ${i}<br>IOPS: ${row[0]}<br>Latency: ${row[1]}<br>ID: ${escapeText(c.id)}<br>block_size: ${escapeText(c.block_size)} | n_jobs: ${escapeText(c.n_jobs)} | iodepth: ${escapeText(c.iodepth)}<br>io_type: ${escapeText(c.io_type)} | raid: ${escapeText(c.raid)} | n_disks: ${escapeText(c.n_disks)}`});
byId('overview').textContent=`Rows: ${rows.length.toLocaleString()} | In initial view: ${source.visible_rows.toLocaleString()} | Outside: ${(rows.length-source.visible_rows).toLocaleString()}`;
for(const name of ['id','raid','block_size','io_type','n_disks','n_jobs','iodepth','read_fraction']) if(configs.some(c=>name in c)){const option=document.createElement('option');option.value=name;option.textContent=name;byId('color-by').append(option)}
function pointColors(){const field=byId('color-by').value;if(field==='cloud'){const mapping=new Map(groups.map(g=>[g.id,g.color]));return labels.map(id=>mapping.get(id)||'#2563eb')};const keys=[...new Set(configs.map(c=>String(c[field])))].sort(), mapping=new Map(keys.map((key,i)=>[key,source.palette[i%source.palette.length]]));return rows.map(row=>mapping.get(String(configs[row[2]][field])))}
function refresh(){const current=Number(byId('cloud').value)||groups[0].id;byId('cloud').replaceChildren();for(const group of groups){const option=document.createElement('option');option.value=group.id;option.textContent=group.name;byId('cloud').append(option)}byId('cloud').value=groups.some(g=>g.id===current)?current:groups[groups.length-1].id;byId('cloud-name').value=groups.find(g=>g.id===Number(byId('cloud').value)).name;
byId('selection-count').textContent=selected.length.toLocaleString();const count=labels.filter(id=>id>0).length;byId('assigned-count').textContent=count.toLocaleString();byId('assign').disabled=!selected.length;byId('unassign').disabled=!selected.length;byId('selected-csv').disabled=!selected.length;byId('clouds-csv').disabled=!count;
byId('cloud-table').replaceChildren();for(const group of groups){const indices=labels.map((id,i)=>id===group.id?i:-1).filter(i=>i>=0), identifiers=new Set(indices.map(i=>configs[rows[i][2]].id??rows[i][2])), tr=document.createElement('tr');tr.innerHTML=`<td><span class="swatch" style="background:${group.color}"></span>${escapeText(group.name)}</td><td>${indices.length}</td><td>${identifiers.size}</td><td>${format(median(indices.map(i=>x[i])))}</td><td>${format(median(indices.map(i=>y[i])))}</td>`;byId('cloud-table').append(tr)}
byId('config-table').replaceChildren();const counts=new Map();for(const index of selected){const code=rows[index][2];counts.set(code,(counts.get(code)||0)+1)}for(const [code,count] of [...counts.entries()].sort((a,b)=>b[1]-a[1]).slice(0,20)){const c=configs[code],tr=document.createElement('tr');tr.innerHTML=[c.id??code,count,c.block_size,c.n_jobs,c.iodepth,c.io_type,c.raid,c.n_disks].map(v=>`<td>${escapeText(v)}</td>`).join('');byId('config-table').append(tr)}
Plotly.restyle(plot,{'marker.color':[pointColors()]});}
function annotations(){return{version:1,source:source.source,source_sha256:source.sha256,rows:rows.length,index_base:0,iops_divisor:source.iops_divisor,latency_divisor:source.latency_divisor,initial_limits:source.limits,clouds:groups.map(g=>({...g,source_rows:labels.map((id,i)=>id===g.id?i:-1).filter(i=>i>=0)}))}}
function download(text,name,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function csvCell(value){const s=String(value??'');return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function exportCsv(indices,name){const columns=['source_row','cloud_id','cloud_name',...source.columns],mapping=new Map(groups.map(g=>[g.id,g.name]));const result=[columns.map(csvCell).join(',')];for(const index of indices){const row=rows[index],c=configs[row[2]],values=[index,labels[index],mapping.get(labels[index])||'',...source.columns.map(column=>column==='iops'?row[0]:column==='lat'?row[1]:c[column])];result.push(values.map(csvCell).join(','))}download(result.join('\r\n')+'\r\n',name,'text/csv;charset=utf-8')}
window.cloudExplorer={getAnnotations:annotations,getSelectedRows:()=>selected.slice()};
Plotly.newPlot(plot,[{type:'scattergl',mode:'markers',x,y,text:hover,hovertemplate:'%{text}<extra></extra>',marker:{size:3,color:'#2563eb',opacity:.45},selected:{marker:{color:'#dc2626',opacity:.9,size:5}},unselected:{marker:{opacity:.16}}}],{margin:{l:75,r:25,t:20,b:70},dragmode:'lasso',hovermode:'closest',uirevision:'ground-truth',xaxis:{title:{text:`IOPS / ${source.iops_divisor}`},range:source.limits.iops,zeroline:false},yaxis:{title:{text:`Latency / ${source.latency_divisor}`},range:source.limits.lat,zeroline:false},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui,sans-serif',size:13,color:'#20252a'}},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'ground_truth_clouds',width:1000,height:800,scale:2}}).then(()=>{
plot.on('plotly_selected',event=>{selected=event?event.points.filter(point=>point.curveNumber===0).map(point=>point.pointIndex):[];geometry=event?.lassoPoints||event?.range||null;refresh()});plot.on('plotly_deselect',()=>{selected=[];geometry=null;refresh()});refresh();
});
byId('cloud').addEventListener('change',()=>{byId('cloud-name').value=groups.find(g=>g.id===Number(byId('cloud').value)).name});
byId('cloud-name').addEventListener('change',()=>{const group=groups.find(g=>g.id===Number(byId('cloud').value));group.name=byId('cloud-name').value.trim()||`Cloud ${group.id}`;refresh()});
byId('new-cloud').addEventListener('click',()=>{const id=Math.max(...groups.map(g=>g.id))+1;groups.push({id,name:`Cloud ${id}`,color:source.palette[(id-1)%source.palette.length]});refresh();byId('cloud').value=id;byId('cloud-name').value=`Cloud ${id}`});
byId('assign').addEventListener('click',()=>{const id=Number(byId('cloud').value);for(const index of selected)labels[index]=id;const group=groups.find(g=>g.id===id);if(geometry)group.last_selection=geometry;refresh()});
byId('unassign').addEventListener('click',()=>{for(const index of selected)labels[index]=0;refresh()});
byId('color-by').addEventListener('change',refresh);
byId('selected-csv').addEventListener('click',()=>exportCsv(selected,'selected_ground_truth_rows.csv'));
byId('clouds-csv').addEventListener('click',()=>exportCsv(labels.map((id,i)=>id>0?i:-1).filter(i=>i>=0),'ground_truth_cloud_rows.csv'));
byId('export-json').addEventListener('click',()=>download(JSON.stringify(annotations(),null,2),'ground_truth_cloud_annotations.json','application/json'));
byId('import-json').addEventListener('change',async event=>{byId('error').textContent='';try{const file=event.target.files[0];if(!file)return;const incoming=JSON.parse(await file.text());if(incoming.source_sha256!==source.sha256||incoming.rows!==rows.length)throw Error('Annotations belong to another CSV');const next=new Array(rows.length).fill(0),identifiers=new Set();for(const group of incoming.clouds){if(!Number.isInteger(group.id)||group.id<=0||identifiers.has(group.id))throw Error('Invalid cloud ID');identifiers.add(group.id);for(const index of group.source_rows){if(!Number.isInteger(index)||index<0||index>=rows.length||next[index])throw Error('Invalid or duplicate source row');next[index]=group.id}}groups=incoming.clouds.map(g=>({id:g.id,name:String(g.name),color:source.palette[(g.id-1)%source.palette.length]}));if(!groups.length)groups=[{id:1,name:'Cloud 1',color:source.palette[0]}];labels=next;refresh()}catch(error){byId('error').textContent=error.message}});
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=Path("dataset/pools/test_hdd_sequential.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ground_truth_clouds/hdd_sequential_test"))
    parser.add_argument("--iops-limits", nargs=2, type=float, default=[0, 14])
    parser.add_argument("--lat-limits", nargs=2, type=float, default=[0, 100])
    parser.add_argument("--iops-divisor", type=float, default=1e3)
    parser.add_argument("--latency-divisor", type=float, default=1e6)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--highlight-ids", nargs="+", help="Highlight all rows of experiment IDs identified in visible clouds.")
    args = parser.parse_args()
    for bounds in (args.iops_limits, args.lat_limits):
        if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Axis limits must be finite and increasing.")
    if not np.isfinite([args.iops_divisor, args.latency_divisor]).all() or min(args.iops_divisor, args.latency_divisor) <= 0:
        raise ValueError("Divisors must be finite and positive.")
    frame = pd.read_csv(args.data_csv, low_memory=False)
    if frame.empty:
        raise ValueError("Source CSV is empty.")
    payload = prepare_payload(frame, args.data_csv, {"iops": args.iops_limits, "lat": args.lat_limits}, args.iops_divisor, args.latency_divisor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels, groups = None, None
    annotation_source = str(args.annotations) if args.annotations else None
    if args.annotations and args.highlight_ids:
        raise ValueError("Choose annotations or highlight-ids, not both.")
    if args.highlight_ids:
        if "id" not in frame:
            raise ValueError("Highlighting experiment IDs requires an id column.")
        missing = set(args.highlight_ids) - set(frame.id)
        if missing:
            raise ValueError(f"Unknown experiment IDs: {sorted(missing)}")
        annotations = {"version": 1, "source_sha256": payload["sha256"], "rows": len(frame), "index_base": 0,
                       "selection_method": "All rows of experiment IDs identified in visible clouds.", "clouds": []}
        for identifier, source_id in enumerate(dict.fromkeys(args.highlight_ids), 1):
            annotations["clouds"].append({"id": identifier, "name": f"Cloud {identifier}", "color": PALETTE[(identifier-1)%len(PALETTE)],
                                         "experiment_id": source_id, "source_rows": np.flatnonzero(frame.id.eq(source_id)).tolist()})
        generated_annotations = args.output_dir / "example_cloud_annotations.json"
        generated_annotations.write_text(json.dumps(annotations, indent=2))
        labels, groups = load_annotations(generated_annotations, payload)
        payload["initial_annotations"] = annotations
        annotation_source = str(generated_annotations)
    elif args.annotations:
        labels, groups = load_annotations(args.annotations, payload)
        payload["initial_annotations"] = json.loads(args.annotations.read_text())
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    html = HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__PAYLOAD__", encoded)
    (args.output_dir / "ground_truth_clouds.html").write_text(html)
    save_plot(frame, payload, args.output_dir / "ground_truth.png")
    frame.assign(source_row=np.arange(len(frame))).to_csv(args.output_dir / "ground_truth_rows.csv", index=False)
    if labels is not None:
        annotated = frame.copy()
        annotated.insert(0, "source_row", np.arange(len(frame)))
        annotated.insert(1, "cloud_id", labels)
        annotated.insert(2, "cloud_name", [groups.get(identifier, {}).get("name", "") for identifier in labels])
        annotated.to_csv(args.output_dir / "all_rows_with_clouds.csv", index=False)
        annotated.loc[labels > 0].to_csv(args.output_dir / "cloud_rows.csv", index=False)
        summary = []
        configuration_rows = []
        for identifier, group in groups.items():
            selected = annotated.loc[labels == identifier]
            selected.to_csv(args.output_dir / f"cloud_{identifier:02d}_rows.csv", index=False)
            summary.append({"cloud_id": identifier, "cloud_name": group["name"], "rows": len(selected),
                            "ids": selected.id.nunique() if "id" in selected else None,
                            "median_iops": selected.iops.median(), "median_lat": selected.lat.median()})
            context_columns = [column for column in frame if column not in {"iops", "lat"}]
            if context_columns and len(selected):
                counts = selected.groupby(context_columns, dropna=False, sort=False).size().rename("rows").reset_index()
                counts.insert(0, "cloud_id", identifier)
                configuration_rows.extend(counts.to_dict("records"))
        pd.DataFrame(summary).to_csv(args.output_dir / "cloud_summary.csv", index=False)
        pd.DataFrame(configuration_rows).to_csv(args.output_dir / "cloud_configurations.csv", index=False)
        save_plot(frame, payload, args.output_dir / "ground_truth_annotated.png", labels, groups)
        save_cloud_panels(frame, payload, args.output_dir / "clouds_local_views.png", labels, groups)
    (args.output_dir / "run_info.json").write_text(json.dumps({
        "source": payload["source"], "source_sha256": payload["sha256"], "rows": len(frame),
        "visible_rows": payload["visible_rows"], "outside_initial_view": len(frame)-payload["visible_rows"],
        "limits": payload["limits"], "iops_divisor": args.iops_divisor, "latency_divisor": args.latency_divisor,
        "annotations": annotation_source,
        "row_index": "Zero-based source CSV row, excluding the header.",
        "selection": "Experiment IDs chosen from visible clouds." if args.highlight_ids else "Manual lasso/box annotations; no inferred cluster labels.",
    }, indent=2))
    print(f"Rows: {len(frame)}; initial view: {payload['visible_rows']}; configurations: {len(payload['configurations'])}")
    print(f"Explorer: {args.output_dir / 'ground_truth_clouds.html'}")


if __name__ == "__main__":
    main()
