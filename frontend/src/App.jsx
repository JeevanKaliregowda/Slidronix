import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, AlertTriangle, BarChart3, Bell, CheckCircle2, ChevronDown,
  CircleHelp, Database, FileText, Gauge, LogOut, Map as MapIcon,
  MapPin, Menu, Mountain, Navigation, Plus, RadioTower, RefreshCw, Route,
  Search, Settings, ShieldAlert, Siren, Users, X, Zap
} from "lucide-react";
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from "react-leaflet";
import { CheckAreaTool, SafeRouteTool } from "./LocationTools";
import "leaflet/dist/leaflet.css";
import "./App.css";
import "./Theme.css";

const API = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const authHeaders = () => {
  const token = localStorage.getItem("slidronix_access_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
};
const RISK = { Low: "#23865a", Moderate: "#c28a16", High: "#df7133", Critical: "#c83d48" };
const ROLE_LABELS = { citizen: "Citizen", rescue: "Rescue team", researcher: "Researcher", admin: "Administrator" };
const NAV = {
  citizen: [{id:"overview",label:"Overview",icon:Gauge},{id:"map",label:"Risk map",icon:MapIcon},{id:"check",label:"Check my area",icon:Navigation},{id:"route",label:"Safe route",icon:Route},{id:"incidents",label:"Report incident",icon:FileText},{id:"sos",label:"SOS requests",icon:Siren},{id:"alerts",label:"Emergency alerts",icon:Bell}],
  rescue: [{id:"overview",label:"Operations overview",icon:Gauge},{id:"map",label:"Risk & incident map",icon:MapIcon},{id:"incidents",label:"Incident queue",icon:AlertTriangle},{id:"sos",label:"SOS response",icon:Siren},{id:"scenario",label:"Simulate scenario",icon:Activity},{id:"route",label:"Safe route",icon:Route},{id:"alerts",label:"Send alert",icon:Bell}],
  researcher: [{id:"overview",label:"Research overview",icon:Gauge},{id:"map",label:"Risk map",icon:MapIcon},{id:"datasets",label:"Datasets",icon:Database},{id:"models",label:"Models & metrics",icon:Activity},{id:"analysis",label:"Risk analysis",icon:BarChart3},{id:"research",label:"Research notes",icon:FileText}],
  admin: [{id:"overview",label:"System overview",icon:Gauge},{id:"incidents",label:"Incident management",icon:AlertTriangle},{id:"sos",label:"SOS management",icon:Siren},{id:"alerts",label:"Alert management",icon:Bell},{id:"users",label:"User management",icon:Users},{id:"data",label:"Data & models",icon:Database},{id:"settings",label:"Settings",icon:Settings}]
};
const safeArray = (x) => Array.isArray(x) ? x : (x?.items || x?.points || []);
const fmt = (v, digits=0) => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined,{maximumFractionDigits:digits}) : "—";
const riskOf = (p) => p?.Risk_Level || p?.risk_level || "Low";
const latOf = (p) => Number(p?.Latitude ?? p?.latitude);
const lonOf = (p) => Number(p?.Longitude ?? p?.longitude);
const titleOf = (x) => x?.title || x?.message || x?.description || `Record #${x?.id ?? "—"}`;

function FlyTo({ point }) { const map = useMap(); useEffect(()=>{if(point && Number.isFinite(latOf(point)) && Number.isFinite(lonOf(point))) map.flyTo([latOf(point),lonOf(point)],10,{duration:.7});},[point,map]); return null; }

function WorkspaceApp({ session, onLogout }) {
  const role = session.role;
  const [page,setPage] = useState("overview");
  const [points,setPoints] = useState([]);
  const [incidents,setIncidents] = useState([]);
  const [sosItems,setSosItems] = useState([]);
  const [alerts,setAlerts] = useState([]);
  const [users,setUsers] = useState([]);
  const [selected,setSelected] = useState(null);
  const [filter,setFilter] = useState("All");
  const [search,setSearch] = useState("");
  const [busy,setBusy] = useState(false);
  const [notice,setNotice] = useState("");
  const [error,setError] = useState("");
  const [loading,setLoading] = useState(true);
  const [incidentForm,setIncidentForm] = useState({title:"",description:"",category:"landslide",latitude:"13.3161",longitude:"75.772",reporter_name:session.name});
  const [sosForm,setSosForm] = useState({message:"Emergency assistance requested",latitude:"13.3161",longitude:"75.772",contact_name:session.name,contact_phone:""});
  const [alertForm,setAlertForm] = useState({title:"",message:"",risk_level:"High",latitude:"13.3161",longitude:"75.772"});
  const [mobileNav,setMobileNav] = useState(false);
  const [notificationPermission,setNotificationPermission] = useState(
    typeof Notification === "undefined" ? "unsupported" : Notification.permission
  );
  const seenAlertIds = useRef(null);

  const api = useCallback(async (path,options={})=>{
    const response = await fetch(`${API}${path}`,{...options,headers:{...authHeaders(),"Content-Type":"application/json",...(options.headers||{})}});
    const data = await response.json().catch(()=>({}));
    if(response.status===401) window.dispatchEvent(new Event("slidronix-auth-expired"));
    if(!response.ok) throw new Error(typeof data.detail==="string"?data.detail:`Request failed (${response.status})`);
    return data;
  },[]);

  const refreshOperations = useCallback(async ()=>{
    const canReadQueues = ["citizen","rescue","admin"].includes(role);
    const requests = await Promise.allSettled([
      canReadQueues ? api("/incidents?limit=100") : Promise.resolve([]),
      canReadQueues ? api("/sos?limit=100") : Promise.resolve([]),
      api("/alerts?limit=100"),
      role === "admin" ? api("/auth/admin/users") : Promise.resolve([]),
    ]);
    if(requests[0].status==="fulfilled") setIncidents(safeArray(requests[0].value));
    if(requests[1].status==="fulfilled") setSosItems(safeArray(requests[1].value));
    if(requests[2].status==="fulfilled") setAlerts(safeArray(requests[2].value));
    if(requests[3].status==="fulfilled") setUsers(safeArray(requests[3].value));
    const critical = role === "admin" ? requests.slice(2) : role === "researcher" ? [requests[2]] : requests.slice(0,3);
    const failed = critical.filter(r=>r.status==="rejected");
    if(failed.length===critical.length) setError("Operations API is unavailable. Start the FastAPI backend on port 8000.");
    else if(failed.length) setError("Some operational records could not be loaded. Check the backend endpoints.");
    else setError("");
  },[api,role]);

  useEffect(()=>{
    let cancelled=false;
    async function load(){
      setLoading(true);
      try {
        const [summaryRes] = await Promise.all([fetch(`${API}/predictions/summary`,{headers:authHeaders()})]);
        if(!summaryRes.ok){if(summaryRes.status===401)window.dispatchEvent(new Event("slidronix-auth-expired"));throw new Error("Prediction summary unavailable");}
        const summary=await summaryRes.json(), all=[];
        for(let offset=0;offset<(summary.total_locations||0);offset+=1000){
          const res=await fetch(`${API}/predictions/points?limit=1000&offset=${offset}`,{headers:authHeaders()});
          if(!res.ok){if(res.status===401)window.dispatchEvent(new Event("slidronix-auth-expired"));throw new Error("Prediction points unavailable");}
          const data=await res.json(); all.push(...(data.points||[]));
        }
        if(!cancelled) setPoints(all.map(p=>({ID:`${p.latitude}_${p.longitude}`,Latitude:p.latitude,Longitude:p.longitude,Landslide_Probability:p.landslide_probability,Risk_Level:p.risk_level})));
      } catch {
        try { const res=await fetch("/data/risk_predictions.json"); if(res.ok){const d=await res.json();if(!cancelled)setPoints(Array.isArray(d)?d:(d.points||[]));} }
        catch { /* API status below communicates the unavailable dataset. */ }
      } finally { if(!cancelled) setLoading(false); }
    }
    load();
    Promise.resolve().then(()=>{if(!cancelled)refreshOperations();});
    return ()=>{cancelled=true;};
  },[refreshOperations]);

  // Foreground mobile/browser warning simulation. Existing alerts form the baseline,
  // so old warnings do not trigger a burst of notifications on page load.
  useEffect(()=>{
    let cancelled=false;
    async function checkForWarnings(){
      try{
        const latest=safeArray(await api("/alerts?limit=100"));
        if(cancelled)return;
        if(seenAlertIds.current===null){
          seenAlertIds.current=new Set(latest.map(item=>item.id));
          setAlerts(latest);
          return;
        }
        const previous=seenAlertIds.current;
        const newWarnings=latest.filter(item=>
          !previous.has(item.id) && ["high","critical"].includes(String(item.risk_level||"").toLowerCase())
          && !["resolved","cancelled","closed"].includes(String(item.status||"active").toLowerCase())
        );
        latest.forEach(item=>previous.add(item.id));
        setAlerts(latest);
        if(typeof Notification!=="undefined" && Notification.permission==="granted"){
          newWarnings.forEach(item=>{
            const risk=String(item.risk_level||"High").toUpperCase();
            new Notification("Slidronix "+risk+" landslide warning",{
              body:item.title+(item.message?": "+item.message:""),
              tag:"slidronix-alert-"+item.id,
              icon:"/favicon.ico"
            });
          });
        }
      }catch{
        // Keep the dashboard usable if polling briefly fails.
      }
    }
    checkForWarnings();
    const timer=window.setInterval(checkForWarnings,15000);
    return ()=>{cancelled=true;window.clearInterval(timer);};
  },[api]);

  async function enableBrowserNotifications(){
    if(typeof Notification==="undefined"){
      setError("This browser does not support web notifications. Try a supported mobile browser over HTTPS.");
      return;
    }
    try{
      const permission=await Notification.requestPermission();
      setNotificationPermission(permission);
      if(permission==="granted"){
        setNotice("Notifications enabled on this device while Slidronix is open.");
        new Notification("Slidronix notifications enabled",{body:"High and Critical landslide warnings will appear while this dashboard remains open."});
      }else if(permission==="denied"){
        setError("Notifications are blocked for this site. Change the site's notification permission in your browser settings.");
      }else{
        setNotice("Notification permission was not granted.");
      }
    }catch{
      setError("Could not request notification permission in this browser.");
    }
  }

  const counts=useMemo(()=>({
    total:points.length,low:points.filter(p=>riskOf(p)==="Low").length,
    moderate:points.filter(p=>riskOf(p)==="Moderate").length,
    high:points.filter(p=>["High","Critical"].includes(riskOf(p))).length,
    activeIncidents:incidents.filter(i=>!["resolved","closed"].includes(String(i.status).toLowerCase())).length,
    openSos:sosItems.filter(s=>!["resolved","cancelled"].includes(String(s.status).toLowerCase())).length,
    activeAlerts:alerts.filter(a=>!(["resolved","cancelled","closed"].includes(String(a.status||"active").toLowerCase()))).length
  }),[points,incidents,sosItems,alerts]);
  const visiblePoints=useMemo(()=>points.filter(p=>(filter==="All"||riskOf(p)===filter)&&(!search||(String(p.Latitude).includes(search)||String(p.Longitude).includes(search)))),[points,filter,search]);

  async function submitRecord(path,payload,label,reset){
    setBusy(true);setNotice("");setError("");
    try{await api(path,{method:"POST",body:JSON.stringify(payload)});setNotice(`${label} saved successfully.`);reset();await refreshOperations();}
    catch(e){setError(`${label} could not be saved: ${e.message}`);}
    finally{setBusy(false);}
  }
  async function updateStatus(kind,id,status){
    setError("");setNotice("");
    try{await api(`/${kind}/${id}/status`,{method:"PATCH",body:JSON.stringify({status})});setNotice("Status updated.");await refreshOperations();}
    catch(e){setError(`Status update failed: ${e.message}`);}
  }
  function captureBrowserLocation(setLat,setLon){
    if(!navigator.geolocation){setError("Browser geolocation is not supported.");return;}
    navigator.geolocation.getCurrentPosition(pos=>{setLat(String(pos.coords.latitude.toFixed(5)));setLon(String(pos.coords.longitude.toFixed(5)));setNotice("Current coordinates captured.");},err=>setError(`Location unavailable: ${err.message}`),{enableHighAccuracy:true,timeout:10000});
  }
  function navTo(id){setPage(id);setMobileNav(false);setNotice("");setError("");}
  const roleNav=NAV[role]||NAV.citizen;
  const pageTitle=roleNav.find(n=>n.id===page)?.label||"Overview";

  const field=(label,value,onChange,props={})=><label className="field" key={label}><span>{label}</span><input value={value} onChange={e=>onChange(e.target.value)} {...props}/></label>;
  const coords=(form,setForm)=><div className="field-grid">{field("Latitude",form.latitude,v=>setForm(o=>({...o,latitude:v})),{type:"number",step:"any",required:true})}{field("Longitude",form.longitude,v=>setForm(o=>({...o,longitude:v})),{type:"number",step:"any",required:true})}</div>;
  const submitButton=(label)=><button className="btn btn-primary" disabled={busy} type="submit">{busy?<RefreshCw size={16} className="spin"/>:<Plus size={16}/>} {busy?"Saving…":label}</button>;
  const statusSelect=(kind,item,options)=><select className="status-select" value={item.status||options[0][0]} onChange={e=>updateStatus(kind,item.id,e.target.value)}>{options.map(([v,l])=><option key={v} value={v}>{l}</option>)}</select>;

  function recordList(items,kind){
    if(!items.length)return <div className="empty-state"><FileText size={25}/><strong>No records yet</strong><span>Submitted records will appear here after they are saved.</span></div>;
    return <div className="record-list">{items.map(item=><article className="record" key={`${kind}-${item.id}`}><div className="record-head"><div><span className="record-id">#{item.id} · {kind==="sos"?"SOS REQUEST":kind==="alerts"?"ALERT":"INCIDENT"}</span><h3>{titleOf(item)}</h3></div><span className={`pill ${String(item.risk_level||item.severity||item.status||"open").toLowerCase().replace(/\s/g,"-")}`}>{item.status||item.risk_level||item.severity||"Active"}</span></div>{item.description&&<p>{item.description}</p>}{item.category&&<p className="muted">{item.category}</p>}<div className="record-foot"><span><MapPin size={13}/>{Number.isFinite(latOf(item))?`${latOf(item).toFixed(4)}, ${lonOf(item).toFixed(4)}`:"Location not supplied"}</span><span>{item.created_at?new Date(item.created_at).toLocaleString():""}</span></div>{kind==="incidents"&&<div className="record-actions"><label>Update status</label>{statusSelect("incidents",item,[["reported","Reported"],["under_review","Under review"],["responding","Responding"],["resolved","Resolved"],["closed","Closed"]])}</div>}{kind==="sos"&&<div className="record-actions"><label>Response status</label>{statusSelect("sos",item,[["open","Open"],["acknowledged","Acknowledged"],["dispatched","Dispatched"],["resolved","Resolved"],["cancelled","Cancelled"]])}</div>}</article>)}</div>;
  }

  function RiskMap({height=470,showIncidents=false}){
    return <div className="map-frame" style={{height}}><MapContainer center={[13.05,75.7]} zoom={7} scrollWheelZoom className="risk-map"><TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/><FlyTo point={selected}/>{visiblePoints.filter(p=>Number.isFinite(latOf(p))&&Number.isFinite(lonOf(p))).map((p,i)=><CircleMarker key={p.ID||i} center={[latOf(p),lonOf(p)]} radius={riskOf(p)==="High"?5:4} pathOptions={{color:RISK[riskOf(p)]||RISK.Low,fillColor:RISK[riskOf(p)]||RISK.Low,fillOpacity:.62,weight:1}} eventHandlers={{click:()=>setSelected(p)}}><Popup><strong>{riskOf(p)} estimated risk</strong><br/>Probability: {fmt((p.Landslide_Probability||0)*100,2)}%<br/>{latOf(p).toFixed(4)}, {lonOf(p).toFixed(4)}</Popup></CircleMarker>)}{showIncidents&&[...incidents,...sosItems].filter(p=>Number.isFinite(latOf(p))&&Number.isFinite(lonOf(p))).map((p,i)=><CircleMarker key={`ops-${p.id}-${i}`} center={[latOf(p),lonOf(p)]} radius={8} pathOptions={{color:"#fff",weight:2,fillColor:p.message?"#c83d48":"#315fc4",fillOpacity:.95}}><Popup><strong>{p.message?"SOS request":"Incident"} #{p.id}</strong><br/>{titleOf(p)}<br/>Status: {p.status}</Popup></CircleMarker>)}</MapContainer><div className="map-legend"><span><i style={{background:RISK.Low}}/>Low</span><span><i style={{background:RISK.Moderate}}/>Moderate</span><span><i style={{background:RISK.High}}/>High</span>{showIncidents&&<span><i style={{background:"#c83d48"}}/>SOS / incident</span>}</div></div>;
  }

  function Stat({label,value,sub,icon:Icon,tone="blue"}){return <div className="stat-card"><div className={`stat-icon ${tone}`}><Icon size={19}/></div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-sub">{sub}</div></div>;}
  function SectionHead({eyebrow,title,description,action}){return <div className="section-head"><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2>{description&&<p>{description}</p>}</div>{action}</div>;}
  function MapWorkspace(){return <><SectionHead eyebrow="GEOSPATIAL INTELLIGENCE" title="Landslide risk map" description="Explore modelled risk locations and inspect incident locations on the same map." action={<button className="btn btn-secondary" onClick={refreshOperations}><RefreshCw size={15}/> Refresh data</button>}/><div className="toolbar"><label className="searchbox"><Search size={16}/><input placeholder="Search coordinates…" value={search} onChange={e=>setSearch(e.target.value)}/></label><select value={filter} onChange={e=>setFilter(e.target.value)} className="toolbar-select"><option>All</option><option>Low</option><option>Moderate</option><option>High</option></select><span className="result-count">{fmt(visiblePoints.length)} map points</span></div>{RiskMap({height:560,showIncidents:role!=="researcher"})}<div className="map-footnote">{InfoIcon()} Risk estimates are not guarantees of an event or its timing. Some environmental inputs may use reference samples.</div></>;}
  function CheckArea(){return <><SectionHead eyebrow="LOCATION ASSESSMENT" title="Check my area" description="Choose a point on the map or search for a place, then run the existing Fusion coordinate inference."/><CheckAreaTool api={api} onAssessment={(result)=>setSelected({Latitude:result.latitude,Longitude:result.longitude,Risk_Level:result.risk_level,Landslide_Probability:result.landslide_probability,isFreshInference:true})}/></>;}
  function IncidentForm(){return <><SectionHead eyebrow="COMMUNITY REPORTING" title="Report an incident" description="Record a landslide or other observed hazard for operational review."/><div className="split-layout"><section className="surface-card"><form onSubmit={e=>{e.preventDefault();submitRecord("/incidents",{...incidentForm,latitude:Number(incidentForm.latitude),longitude:Number(incidentForm.longitude)},"Incident",()=>setIncidentForm({title:"",description:"",category:"landslide",latitude:"13.3161",longitude:"75.772",reporter_name:session.name}));}}><h3><FileText size={18}/> Incident details</h3>{field("Short title",incidentForm.title,v=>setIncidentForm(o=>({...o,title:v})),{required:true,placeholder:"e.g. Rockfall near road"})}<label className="field"><span>Hazard type</span><select value={incidentForm.category} onChange={e=>setIncidentForm(o=>({...o,category:e.target.value}))}><option value="landslide">Landslide</option><option value="rockfall">Rockfall</option><option value="road_block">Road blockage</option><option value="flooding">Flooding</option><option value="fallen_tree">Fallen tree</option><option value="damage">Structural damage</option><option value="other">Other</option></select></label><label className="field"><span>Description</span><textarea rows="3" value={incidentForm.description} onChange={e=>setIncidentForm(o=>({...o,description:e.target.value}))} placeholder="Describe what you observed…"/></label>{coords(incidentForm,setIncidentForm)}<div className="form-actions"><button type="button" className="btn btn-secondary" onClick={()=>captureBrowserLocation(v=>setIncidentForm(o=>({...o,latitude:v})),v=>setIncidentForm(o=>({...o,longitude:v}))) }><MapPin size={15}/> Capture location</button>{submitButton("Submit report")}</div></form></section><section className="surface-card"><h3><MapIcon size={18}/> Recent reports</h3>{recordList(incidents.slice(0,5),"incidents")}</section></div></>;}
  function SosWorkspace(){const citizen=role==="citizen";return <><SectionHead eyebrow="EMERGENCY RESPONSE" title={citizen?"Request assistance":"SOS response queue"} description={citizen?"Submit a project demonstration request. This does not contact emergency services.":"Review incoming requests and update response progress."}/><div className="split-layout"><section className="surface-card"><form onSubmit={e=>{e.preventDefault();submitRecord("/sos",{...sosForm,latitude:Number(sosForm.latitude),longitude:Number(sosForm.longitude)},"SOS request",()=>setSosForm({message:"Emergency assistance requested",latitude:"13.3161",longitude:"75.772",contact_name:session.name,contact_phone:""}));}}><h3><Siren size={18}/> SOS request</h3><label className="field"><span>Message</span><textarea rows="3" required value={sosForm.message} onChange={e=>setSosForm(o=>({...o,message:e.target.value}))}/></label>{coords(sosForm,setSosForm)}{field("Contact name",sosForm.contact_name,v=>setSosForm(o=>({...o,contact_name:v})),{required:true})}{field("Contact phone (optional)",sosForm.contact_phone,v=>setSosForm(o=>({...o,contact_phone:v})))}<div className="form-actions"><button type="button" className="btn btn-secondary" onClick={()=>captureBrowserLocation(v=>setSosForm(o=>({...o,latitude:v})),v=>setSosForm(o=>({...o,longitude:v})))}><MapPin size={15}/> Capture location</button>{submitButton("Submit SOS request")}</div><p className="helper-text danger-text">Prototype only: no dispatch, SMS, or emergency service is contacted.</p></form></section><section className="surface-card"><h3><RadioTower size={18}/> {citizen?"Your requests":"SOS operations"}</h3>{recordList(sosItems,"sos")}</section></div></>;}
  function AlertsWorkspace(){return <><SectionHead eyebrow="ALERT CENTRE" title={role==="rescue"||role==="admin"?"Publish an alert":"Emergency alerts"} description="High and Critical warnings are checked every 15 seconds while this dashboard is open."/><section className="surface-card" style={{marginBottom:16}}><div className="record-head"><div><h3><Bell size={18}/> Mobile warning notifications</h3><p className="muted">Allow notifications on this device to receive newly published High or Critical alerts while Slidronix is open. This is a demo simulation, not an official emergency warning service.</p></div><button className="btn btn-primary" type="button" onClick={enableBrowserNotifications} disabled={notificationPermission==="granted"||notificationPermission==="unsupported"}>{notificationPermission==="granted"?"Notifications enabled":notificationPermission==="denied"?"Permission blocked":notificationPermission==="unsupported"?"Not supported":"Enable notifications"}</button></div>{notificationPermission==="denied"&&<p className="helper-text danger-text">Notifications are blocked by your browser. Allow them in site settings, then reload this page.</p>}{notificationPermission==="unsupported"&&<p className="helper-text">This browser does not expose the Notification API.</p>}</section><div className="split-layout">{(role==="rescue"||role==="admin")&&<section className="surface-card"><form onSubmit={e=>{e.preventDefault();submitRecord("/alerts",{...alertForm,latitude:alertForm.latitude===""?null:Number(alertForm.latitude),longitude:alertForm.longitude===""?null:Number(alertForm.longitude)},"Alert",()=>setAlertForm({title:"",message:"",risk_level:"High",latitude:"13.3161",longitude:"75.772"}));}}><h3><Bell size={18}/> Create alert</h3>{field("Alert title",alertForm.title,v=>setAlertForm(o=>({...o,title:v})),{required:true})}<label className="field"><span>Message</span><textarea rows="3" value={alertForm.message} onChange={e=>setAlertForm(o=>({...o,message:e.target.value}))} required/></label><label className="field"><span>Risk level</span><select value={alertForm.risk_level} onChange={e=>setAlertForm(o=>({...o,risk_level:e.target.value}))}><option>Low</option><option>Moderate</option><option>High</option><option>Critical</option></select></label>{coords(alertForm,setAlertForm)}{submitButton("Publish alert")}</form></section>}<section className="surface-card"><h3><Bell size={18}/> Saved alerts</h3>{alerts.length?recordList(alerts,"alerts"):<div className="empty-state"><Bell size={24}/><strong>No alerts published</strong><span>Published alerts will appear here.</span></div>}</section></div></>;}
  function Overview(){return <><div className="welcome-row"><div><div className="eyebrow">SLIDRONIX / {ROLE_LABELS[role].toUpperCase()}</div><h2>{role==="citizen"?"Your landslide safety overview":role==="rescue"?"Emergency operations":role==="researcher"?"Research workspace":"System administration"}</h2><p>{role==="citizen"?"Understand local risk and take informed next steps.":role==="rescue"?"Monitor reported hazards and coordinate response status.":role==="researcher"?"Inspect the current modelled risk surface and research assets.":"Review operational records and platform data."}</p></div><button className="btn btn-secondary" onClick={()=>{refreshOperations();setNotice("Refreshing operational records…");}}><RefreshCw size={15}/> Refresh</button></div>
    {alerts.length>0&&role!=="researcher"&&<div className="alert-banner"><span className="alert-icon"><ShieldAlert size={19}/></span><div><strong>{alerts.length} saved alert{alerts.length===1?"":"s"}</strong><span>Review the latest alert messages in the alert centre.</span></div><button className="text-button" onClick={()=>navTo("alerts")}>View alerts <ChevronDown size={15}/></button></div>}
    <div className="stats-grid">{role==="citizen"?<><Stat label="Mapped locations" value={fmt(counts.total)} sub="Prepared risk points" icon={MapIcon}/><Stat label="High-risk locations" value={fmt(counts.high)} sub="Requires additional awareness" icon={AlertTriangle} tone="orange"/><Stat label="Open reports" value={fmt(counts.activeIncidents)} sub="Not yet resolved" icon={FileText} tone="purple"/><Stat label="SOS requests" value={fmt(counts.openSos)} sub="Unresolved demo requests" icon={Siren} tone="red"/></>:role==="rescue"?<><Stat label="Open incidents" value={fmt(counts.activeIncidents)} sub="Needs review or response" icon={AlertTriangle} tone="orange"/><Stat label="Open SOS" value={fmt(counts.openSos)} sub="Pending resolution" icon={Siren} tone="red"/><Stat label="Active alerts" value={fmt(counts.activeAlerts)} sub="Stored alert records" icon={Bell} tone="purple"/><Stat label="Mapped risk points" value={fmt(counts.total)} sub="Modelled locations" icon={MapIcon}/></>:role==="researcher"?<><Stat label="Prediction points" value={fmt(counts.total)} sub="Available risk surface" icon={MapIcon}/><Stat label="Low risk" value={fmt(counts.low)} sub={`${counts.total?((counts.low/counts.total)*100).toFixed(1):0}% of locations`} icon={CheckCircle2} tone="green"/><Stat label="Moderate risk" value={fmt(counts.moderate)} sub={`${counts.total?((counts.moderate/counts.total)*100).toFixed(1):0}% of locations`} icon={Activity} tone="orange"/><Stat label="High risk" value={fmt(counts.high)} sub={`${counts.total?((counts.high/counts.total)*100).toFixed(1):0}% of locations`} icon={AlertTriangle} tone="red"/></>:<><Stat label="Risk locations" value={fmt(counts.total)} sub="Prediction dataset" icon={MapIcon}/><Stat label="Open incidents" value={fmt(counts.activeIncidents)} sub="Operational queue" icon={FileText} tone="orange"/><Stat label="Unresolved SOS" value={fmt(counts.openSos)} sub="Response queue" icon={Siren} tone="red"/><Stat label="Saved alerts" value={fmt(counts.activeAlerts)} sub="Alert records" icon={Bell} tone="purple"/></>}</div>
    <div className="overview-grid"><section className="surface-card map-card"><div className="card-heading"><div><span className="eyebrow">PREPARED RISK DATA</span><h3>Western Ghats risk map</h3></div><button className="text-button" onClick={()=>navTo("map")}>Open full map <Navigation size={15}/></button></div>{RiskMap({height:360,showIncidents:role!=="researcher"})}</section><section className="surface-card quick-card"><div className="card-heading"><div><span className="eyebrow">WORKSPACE</span><h3>Quick actions</h3></div><Zap size={18}/></div>{(role==="citizen"?[["check","Check my area","Run a coordinate risk estimate",Navigation],["incidents","Report an incident","Record a hazard",FileText],["sos","Request assistance","Create a demo SOS request",Siren],["route","Plan a route","Search places for road geometry",Route]]:role==="rescue"?[["incidents","Review incidents","Update status of reports",AlertTriangle],["sos","Review SOS","Manage response status",Siren],["alerts","Publish an alert","Store a warning message",Bell],["scenario","Simulate scenario","Run coordinate inference",Activity]]:role==="researcher"?[["datasets","Explore datasets","Inspect known data sources",Database],["models","Model metrics","Review evaluation results",Activity],["analysis","Risk analysis","View risk distribution",BarChart3],["map","Open risk map","Inspect spatial patterns",MapIcon]]:[["incidents","Manage incidents","Review operational records",FileText],["alerts","Manage alerts","Publish or review alerts",Bell],["users","User management","Assign registered account roles",Users],["data","Data & models","Review project metadata",Database]]).map(([id,label,desc,Icon])=><button className="quick-action" key={id} onClick={()=>navTo(id)}><span className="quick-icon"><Icon size={18}/></span><span><strong>{label}</strong><small>{desc}</small></span><Navigation size={15} className="quick-arrow"/></button>)}</section></div>
    {role!=="researcher"&&<section className="surface-card recent-card"><div className="card-heading"><div><span className="eyebrow">LATEST ACTIVITY</span><h3>Recent incidents</h3></div><button className="text-button" onClick={()=>navTo("incidents")}>View queue <Navigation size={15}/></button></div>{recordList(incidents.slice(0,4),"incidents")}</section>}</>;}

  function RiskAnalysis(){const total=counts.total||1;return <><SectionHead eyebrow="MODEL OUTPUTS" title="Risk distribution" description="Distribution of the prepared modelled points. This describes the current dataset, not a forecast of future landslides."/><div className="stats-grid"><Stat label="All points" value={fmt(counts.total)} sub="Loaded from API or static fallback" icon={Database}/><Stat label="Low" value={fmt(counts.low)} sub={`${(counts.low/total*100).toFixed(1)}%`} icon={CheckCircle2} tone="green"/><Stat label="Moderate" value={fmt(counts.moderate)} sub={`${(counts.moderate/total*100).toFixed(1)}%`} icon={Activity} tone="orange"/><Stat label="High" value={fmt(counts.high)} sub={`${(counts.high/total*100).toFixed(1)}%`} icon={AlertTriangle} tone="red"/></div><section className="surface-card distribution-card"><h3>Risk class distribution</h3>{[["Low",counts.low],["Moderate",counts.moderate],["High",counts.high]].map(([label,n])=><div className="bar-row" key={label}><div><span>{label}</span><strong>{fmt(n)}</strong></div><div className="bar-track"><i style={{width:`${n/total*100}%`,background:RISK[label]}}/></div></div>)}</section><div className="surface-card"><h3>Model evaluation reference</h3><div className="metric-grid">{[["ROC-AUC","0.9122"],["Accuracy","0.8218"],["Precision","0.8033"],["Recall","0.8543"],["F1 score","0.8280"]].map(([k,v])=><div key={k}><span>{k}</span><strong>{v}</strong></div>)}</div><p className="helper-text">Evaluation metrics are from the existing project context; they are not a guarantee of operational performance.</p></div></>;}
  function DatasetPage(){return <><SectionHead eyebrow="RESEARCH ASSETS" title="Datasets & environmental inputs" description="Available project data sources and model input families."/><div className="data-grid">{[["Historical landslide inventory","34,207 records","Historical event inventory processed from a 582-page source."],["Integrated modelling dataset","13,982 samples","Balanced spatial samples with separated train/validation/test splits."],["Terrain","Elevation + slope","Copernicus DEM-derived terrain and 32×32 spatial patches."],["Rainfall","30-day sequence","CHIRPS historical data for training; live inference fetches recent daily precipitation."],["Satellite vegetation","NDVI statistics","MODIS NDVI reference features."],["Land cover & soil","Static covariates","MODIS land cover and SoilGrids-derived soil attributes."]].map(([a,b,c])=><article className="surface-card data-item" key={a}><span className="data-icon"><Database size={18}/></span><span className="eyebrow">DATASET</span><h3>{a}</h3><strong>{b}</strong><p>{c}</p></article>)}</div><p className="helper-text">Metadata displayed from the project implementation plan and available model context; this page does not upload or modify research files.</p></>;}
  function ModelsPage(){return <><SectionHead eyebrow="MODEL REGISTRY" title="Models & evaluation" description="Existing model components and recorded evaluation metrics."/><div className="data-grid">{[["Fusion model","Multimodal","Tabular MLP + spatial CNN + rainfall Transformer."],["Random Forest","Tree ensemble","Existing baseline for structured environmental features."],["Tabular MLP","Neural network","Processes 26 tabular environmental features."],["Spatial CNN","Terrain patches","Processes two-channel 32×32 terrain patches."],["Rainfall Transformer","Sequence model","Processes a 30-day rainfall sequence."]].map(([a,b,c])=><article className="surface-card data-item" key={a}><span className="data-icon"><Activity size={18}/></span><span className="eyebrow">{b}</span><h3>{a}</h3><p>{c}</p></article>)}</div>{RiskAnalysis()}</>;}
  function RoutePage(){return <><SectionHead eyebrow="ROUTE PLANNING" title="Route between places" description="Search for real locations and request road geometry from the configured routing provider."/><SafeRouteTool api={api}/></>;}
  function AdminUsers(){return <><SectionHead eyebrow="ACCESS CONTROL" title="User management" description="Assign workspace access to registered accounts."/><div className="surface-card"><div className="record-list">{users.map(account=><article className="record" key={account.id}><div className="record-head"><div><span className="record-id">ACCOUNT #{account.id}</span><h3>{account.name}</h3><p>{account.email}</p></div><select className="status-select" aria-label={`Role for ${account.email}`} value={account.role} disabled={account.id===session.id} onChange={async e=>{try{await api(`/auth/admin/users/${account.id}/role`,{method:"PATCH",body:JSON.stringify({role:e.target.value})});setNotice("Account role updated.");await refreshOperations();}catch(err){setError(`Role update failed: ${err.message}`);}}}><option value="citizen">Citizen</option><option value="rescue">Rescue team</option><option value="researcher">Researcher</option><option value="admin">Administrator</option></select></div></article>)}</div></div></>;}
  function SettingsPage(){return <><SectionHead eyebrow="PLATFORM CONFIGURATION" title="Settings & readiness" description="Current prototype integration status."/><div className="readiness-list">{[["Prediction API","/predictions/summary and /predict/coordinate","connected if backend is running"],["Incident & SOS database","SQLite demonstration database","API-backed record workflows"],["Authentication","JWT login and role guards","configured by backend operator"],["Emergency push","PWA push subscription + delivery","not implemented"],["Safe routing","OSRM road geometry + returned maneuver steps","Demo provider; no route-wide hazard score"],["Production database","PostgreSQL + PostGIS","not configured"]].map(([a,b,c],i)=><div className="readiness-row" key={a}><span className="readiness-dot" style={{background:i<3?"#23865a":"#c28a16"}}/><div><strong>{a}</strong><span>{b}</span></div><em>{c}</em></div>)}</div></>;}
  function InfoIcon(){return <CircleHelp size={15}/>;}

  function renderPage(){
    if(page==="overview") return Overview();
    if(page==="map") return MapWorkspace();
    if(page==="check"||page==="scenario") return CheckArea();
    if(page==="incidents") return role==="citizen"||role==="rescue" ? IncidentForm() : <><SectionHead eyebrow="OPERATIONS" title="Incident management" description="Review and update incidents submitted through the shared API."/><section className="surface-card">{recordList(incidents,"incidents")}</section></>;
    if(page==="sos") return SosWorkspace();
    if(page==="alerts") return AlertsWorkspace();
    if(page==="route") return RoutePage();
    if(page==="datasets"||page==="data") return DatasetPage();
    if(page==="models") return ModelsPage();
    if(page==="analysis") return RiskAnalysis();
    if(page==="research") return <><SectionHead eyebrow="PROJECT DOCUMENTATION" title="Research information" description="Summary of the current research system and its known limitations."/><div className="surface-card prose"><h3>Method overview</h3><p>Slidronix combines terrain, rainfall sequence, vegetation, land-cover and soil features in a multimodal model to estimate landslide risk.</p><h3>Interpretation</h3><p>The model produces a probability estimate and Low, Moderate or High risk class. It does not predict guaranteed event timing and is not a certified emergency-warning service.</p><h3>Current limitations</h3><p>Some live-coordinate environmental values use the nearest reference sample. Rainfall/model distribution compatibility and operational performance require further validation.</p></div></>;
    if(page==="users") return AdminUsers();
    if(page==="settings") return SettingsPage();
    return Overview();
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav?"sidebar-open":""}`}><div className="brand-lockup"><span className="brand-mark"><Mountain size={22}/></span><span><strong>SLIDRONIX</strong><small>LANDSLIDE INTELLIGENCE</small></span><button className="mobile-close" onClick={()=>setMobileNav(false)}><X size={18}/></button></div>
      <div className="workspace-label">WORKSPACE</div><div className="role-switcher"><span className="avatar">{ROLE_LABELS[role][0]}</span><div><strong>{ROLE_LABELS[role]}</strong><small>{session.name}</small></div></div>
      <nav className="side-nav">{roleNav.map(({id,label,icon:Icon})=><button key={id} className={page===id?"active":""} onClick={()=>navTo(id)}><Icon size={18}/><span>{label}</span>{id==="sos"&&counts.openSos>0&&<b className="nav-count">{counts.openSos}</b>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="prototype-status"><span className="status-light"/><div><strong>Research prototype</strong><small>Not an operational warning service</small></div></div><button className="side-bottom-link" onClick={()=>navTo(role==="admin"?"settings":"research")}><CircleHelp size={17}/> About Slidronix</button><button className="side-bottom-link" onClick={()=>{onLogout();}}><LogOut size={17}/> Sign out</button></div>
    </aside>
    {mobileNav&&<button className="nav-scrim" aria-label="Close navigation" onClick={()=>setMobileNav(false)}/>}
    <main className="main-shell"><header className="topbar"><div className="topbar-left"><button className="menu-button" onClick={()=>setMobileNav(true)}><Menu size={20}/></button><div className="breadcrumbs">Workspace <span>/</span> <strong>{pageTitle}</strong></div></div><div className="topbar-right"><span className="system-state"><i/> Authenticated workspace</span><button className="icon-button" title="Refresh records" onClick={refreshOperations}><RefreshCw size={17}/></button><div className="profile-chip"><span className="avatar small-avatar">{ROLE_LABELS[role][0]}</span><span><strong>{ROLE_LABELS[role]}</strong><small>Signed in</small></span></div></div></header>
      <div className="page-content">{(notice||error)&&<div className={`toast ${error?"toast-error":"toast-success"}`}><span>{error||notice}</span><button onClick={()=>{setNotice("");setError("");}}><X size={15}/></button></div>}{loading&&<div className="loading-line"><span/> Loading risk dataset…</div>}{renderPage()}<footer className="app-footer"><span>SLIDRONIX · AI-based landslide risk assessment</span><span>Risk is an estimate · Academic prototype</span></footer></div>
    </main>
  </div>;
}

function AuthScreen({ onAuthenticated }) {
  const [registering,setRegistering] = useState(false);
  const [name,setName] = useState("");
  const [email,setEmail] = useState("");
  const [password,setPassword] = useState("");
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (registering) {
        const registration = await fetch(`${API}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, password }),
        });
        const registrationBody = await registration.json().catch(() => ({}));
        if (!registration.ok) throw new Error(registrationBody.detail || `Registration failed (${registration.status})`);
      }
      const login = await fetch(`${API}/auth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ username: email.trim(), password }),
      });
      const loginBody = await login.json().catch(() => ({}));
      if (!login.ok) throw new Error(loginBody.detail || `Sign in failed (${login.status})`);
      localStorage.setItem("slidronix_access_token", loginBody.access_token);
      const profile = await fetch(`${API}/auth/me`, { headers: authHeaders() });
      const account = await profile.json().catch(() => ({}));
      if (!profile.ok) throw new Error(account.detail || "The account session could not be verified.");
      onAuthenticated(loginBody.access_token, account);
    } catch (err) {
      localStorage.removeItem("slidronix_access_token");
      setError(err.message || "The authentication service could not be reached.");
    } finally {
      setBusy(false);
    }
  }

  return <main className="auth-shell"><section className="auth-card">
    <div className="auth-brand"><span className="brand-mark"><Mountain size={22}/></span><span><strong>SLIDRONIX</strong><small>LANDSLIDE INTELLIGENCE</small></span></div>
    <div className="eyebrow">RESEARCH PROTOTYPE</div>
    <h1>{registering ? "Create a citizen account" : "Sign in to your workspace"}</h1>
    <p className="auth-intro">Access risk maps and project workflows using your Slidronix account.</p>
    {error && <div className="toast toast-error" role="alert">{error}</div>}
    <form onSubmit={submit}>
      {registering && <label className="field">Full name<input autoComplete="name" required minLength={2} maxLength={100} value={name} onChange={e=>setName(e.target.value)}/></label>}
      <label className="field">Email<input type="email" autoComplete="email" required value={email} onChange={e=>setEmail(e.target.value)}/></label>
      <label className="field">Password<input type="password" autoComplete={registering?"new-password":"current-password"} required minLength={registering?12:1} maxLength={128} value={password} onChange={e=>setPassword(e.target.value)}/>{registering&&<small>Use at least 12 characters.</small>}</label>
      <button className="btn btn-primary auth-submit" disabled={busy}>{busy?"Please wait…":registering?"Create account":"Sign in"}</button>
    </form>
    <button className="auth-toggle" type="button" onClick={()=>{setRegistering(!registering);setError("");}}> {registering?"Already registered? Sign in":"New to Slidronix? Create a citizen account"}</button>
    <p className="helper-text">Prototype only. SOS requests do not contact emergency services, and map estimates are not official warnings.</p>
  </section></main>;
}

export default function App() {
  const [token,setToken] = useState(()=>localStorage.getItem("slidronix_access_token"));
  const [session,setSession] = useState(null);
  const [checking,setChecking] = useState(Boolean(localStorage.getItem("slidronix_access_token")));
  const [sessionError,setSessionError] = useState("");
  const signOut=useCallback(()=>{
    localStorage.removeItem("slidronix_access_token");
    setToken(null);
    setSession(null);
    setSessionError("");
  },[]);

  useEffect(()=>{
    let active=true;
    if (!token) return ()=>{active=false;};
    fetch(`${API}/auth/me`,{headers:{Authorization:`Bearer ${token}`}})
      .then(async response=>({response,body:await response.json().catch(()=>({}))}))
      .then(({response,body})=>{
        if (!active) return;
        if (response.ok) setSession(body);
        else { localStorage.removeItem("slidronix_access_token"); setToken(null); setSession(null); }
      })
      .catch(()=>{ if(active){setSession(null);setSessionError("Could not verify your session. Check that the Slidronix API is running.");} })
      .finally(()=>{if(active)setChecking(false);});
    return ()=>{active=false;};
  },[token]);

  useEffect(()=>{
    window.addEventListener("slidronix-auth-expired",signOut);
    return ()=>window.removeEventListener("slidronix-auth-expired",signOut);
  },[signOut]);

  if (checking) return <main className="auth-shell"><div className="auth-card">Verifying your session…</div></main>;
  if (!session) return <>{sessionError&&<div className="auth-session-error">{sessionError}</div>}<AuthScreen onAuthenticated={(accessToken,account)=>{setSession(account);setToken(accessToken);setSessionError("");setChecking(false);}}/></>;
  return <WorkspaceApp session={session} onLogout={signOut}/>;
}
