import io,csv,base64
from pathlib import Path
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image,ImageOps,ImageStat,ImageDraw
import torch
from torchvision import models,transforms
ROOT=Path(__file__).resolve().parent.parent
R=ROOT/"results"; D=ROOT/"dataset"; C=["akiec","bcc","bkl","df","mel","nv","vasc"]
app=FastAPI(title="DermAI Milestone 2")
app.mount("/static",StaticFiles(directory=Path(__file__).parent/"static"),name="static")
net=None; model_status="Model not loaded"
try:
 p=R/"mobilenet_v3_small.pth"
 if p.exists():
  net=models.mobilenet_v3_small(weights=None)
  net.classifier[3]=torch.nn.Linear(net.classifier[3].in_features,7)
  sd=torch.load(p,map_location="cpu",weights_only=True)
  if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
  if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
  net.load_state_dict({k.replace("module.",""):v for k,v in sd.items()})
  net.eval(); model_status="MobileNetV3 loaded"
 else: model_status="Model file missing"
except Exception as e: net=None; model_status="Model load issue: "+str(e)
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def dataurl(im):
 b=io.BytesIO(); im.save(b,format="JPEG",quality=88); return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
def metric(fn):
 try:
  with open(R/fn,encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
  vals=[(float(x["F1"]),float(x["Support"])) for x in rows if x.get("Class") in C]
  n=sum(s for _,s in vals)
  return round(sum(v*s for v,s in vals)/n,4) if n else None
 except: return None
@app.get("/",response_class=HTMLResponse)
def home(): return (Path(__file__).parent/"static/index.html").read_text(encoding="utf-8")
@app.get("/api/status")
def status():
 folder=D/"preprocessed_images"
 return {"model_status":model_status,"model_ready":net is not None,"images":sum(1 for x in folder.rglob("*") if x.is_file()) if folder.exists() else 0}
@app.get("/api/metrics")
def metrics():
 fs={"Baseline":"baseline_classification_report.csv","ResNet18 Transfer Learning":"resnet18_classification_report.csv","ResNet18 CNN":"resnet18_comparison_report.csv","MobileNetV3 CNN":"mobilenet_v3_small_report.csv"}
 a=[{"model":k,"weighted_f1":v} for k,f in fs.items() if (v:=metric(f)) is not None]
 return {"models":a,"best_model":max(a,key=lambda x:x["weighted_f1"])["model"] if a else None}
@app.post("/api/analyze")
async def analyze(file:UploadFile=File(...)):
 if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(400,"Upload an image.")
 raw=await file.read()
 if len(raw)>12*1024*1024: raise HTTPException(400,"Image must be under 12 MB.")
 try: im=Image.open(io.BytesIO(raw)).convert("RGB")
 except: raise HTTPException(400,"Cannot read this image.")
 w,h=im.size; gray=ImageOps.grayscale(im); bright=ImageStat.Stat(gray).mean[0]
 flags=[]
 if min(w,h)<100: flags.append("Low dimensions")
 if bright<25: flags.append("Possibly too dark")
 if bright>235: flags.append("Possibly overexposed")
 proc=ImageOps.autocontrast(im.resize((224,224)))
 sm=ImageOps.grayscale(proc.resize((112,112))); px=sm.load()
 pts=[(x,y) for y in range(112) for x in range(112) if px[x,y]<75]
 over=proc.copy(); dr=ImageDraw.Draw(over)
 if pts: dr.rectangle((min(x for x,y in pts)*2,min(y for x,y in pts)*2,max(x for x,y in pts)*2,max(y for x,y in pts)*2),outline="#ffb547",width=3)
 pred={"error":model_status}
 if net is not None:
  try:
   with torch.no_grad():
    pr=torch.softmax(net(tf(im).unsqueeze(0)),1)[0]; i=int(pr.argmax())
    pred={"class":C[i],"confidence":round(float(pr[i])*100,2),"probabilities":[{"class":C[j],"score":round(float(pr[j])*100,2)} for j in range(7)]}
  except Exception as e: pred={"error":str(e)}
 return {"filename":file.filename,"dimensions":f"{w} × {h}","brightness":round(bright,1),"quality":"Review recommended" if flags else "Passed basic checks","flags":flags,"processed":dataurl(proc),"proposal":dataurl(over),"prediction":pred}
