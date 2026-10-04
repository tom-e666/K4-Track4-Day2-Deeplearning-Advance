"""Full lab runner. No test metrics are opened before validation selection is locked."""
from __future__ import annotations
import json, sys, platform, time, hashlib, shutil, subprocess, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch, timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train, dataset, model, inference, benchmark
import eval as ev

ROOT = Path.cwd()
SUB = ROOT / "submissions/2A202602873_thai_phuc_tien"
RUNS = SUB / "runs"
PRED = SUB / "predictions"
CURVES = SUB / "curves"
for folder in [RUNS, PRED, CURVES]:
    folder.mkdir(parents=True, exist_ok=True)
TABLES = {k: [] for k in ["Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"]}
EPOCHS, BATCH = 10, 32

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if hasattr(x,"tolist") else str(x)), encoding="utf-8")

def persist():
    save_json(SUB / "tables.json", TABLES)
    for name, rows in TABLES.items():
        pd.DataFrame(rows).to_csv(SUB / f"{name}.csv", index=False)
    print("PROGRESS", {k:len(v) for k,v in TABLES.items()}, flush=True)

assert torch.cuda.is_available(), "A real CUDA GPU is required for this run"
RUNTIME = {"python":platform.python_version(), "torch":torch.__version__, "timm":timm.__version__,
           "gpu":torch.cuda.get_device_name(0), "cuda":torch.version.cuda, "epochs":EPOCHS,
           "batch_size":BATCH, "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
save_json(SUB / "runtime.json", RUNTIME)
print("RUNTIME", RUNTIME, flush=True)
print(subprocess.check_output(["nvidia-smi"],text=True),flush=True)

# Use the author's unmodified fold-0 CSVs, not a newly randomized split.
labels = ROOT / "data/labels"
labels.mkdir(parents=True,exist_ok=True)
for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
    url = f"https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/{name}.csv"
    urllib.request.urlretrieve(url, labels / f"{name}.csv")
images = list(Path("/kaggle/input").rglob("20160928-140314-0.jpg"))
assert len(images)==1, f"Expected one DeepWeeds image root; found {images}"
images_dir = images[0].parent
tr,va,te = dataset.load_split(labels)
stats = dataset.check_split(tr,va,te,images_dir)
stats["csv_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in labels.glob("*.csv")}
assert all(abs(n/17509-r)<0.01 for n,r in [(len(tr),.6),(len(va),.2),(len(te),.2)])
save_json(SUB / "split_checks.json", stats)
print("SPLIT_CHECKS", stats,flush=True)
counts=pd.concat([tr,va,te])["Label"].value_counts().sort_index()
plt.figure(figsize=(10,4));plt.bar(dataset.CLASS_NAMES,counts);plt.xticks(rotation=35,ha="right")
plt.ylabel("Images");plt.tight_layout();plt.savefig(CURVES/"class_distribution.png");plt.close()
from PIL import Image
fig,axes=plt.subplots(9,3,figsize=(8,20))
for label in range(9):
    for j,(_,row) in enumerate(tr[tr.Label==label].head(3).iterrows()):
        with Image.open(images_dir/row.Filename) as image:
            axes[label,j].imshow(image)
        axes[label,j].axis("off")
        if j==0: axes[label,j].set_title(dataset.CLASS_NAMES[label])
fig.tight_layout();fig.savefig(CURVES/"sample_classes.png");plt.close(fig)

# Real-image pipeline check uses the same train_one_epoch as the experiments.
train.set_seed(0)
small=tr.groupby("Label",group_keys=False).head(1).reset_index(drop=True)
loader=dataset.make_loader(small,images_dir,dataset.build_transforms(False),9,False,num_workers=0)
x,y,names=next(iter(loader)); x=x.cuda(); y=y.cuda()
net=model.build_model("resnet50",True).cuda()
criterion=torch.nn.CrossEntropyLoss()
net.eval()
with torch.no_grad(): initial=float(criterion(net(x),y))
assert abs(initial-np.log(9))<1.0, f"Unexpected initial loss: {initial}"
opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
for step in range(100):
    result=train.train_one_epoch(net,[(x,y,names)],criterion,opt,None,None,train.Config(amp=False),torch.device("cuda"))
    if result["train_loss"]<.05: break
assert result["train_loss"]<.05, result
save_json(SUB/"pipeline_checks.json",{"initial_ce":initial,"expected_ce":float(np.log(9)),
    "overfit_train_loss":result["train_loss"],"overfit_steps":step+1,"real_images":names})
aug_loader=dataset.make_loader(small,images_dir,dataset.build_transforms(True),9,True,num_workers=0)
ax_img,ax_y,_=next(iter(aug_loader))
fig,axes=plt.subplots(3,3,figsize=(9,9))
for i,ax in enumerate(axes.flat):
    image=ax_img[i]*torch.tensor(dataset.IMAGENET_STD)[:,None,None]+torch.tensor(dataset.IMAGENET_MEAN)[:,None,None]
    ax.imshow(image.clamp(0,1).permute(1,2,0));ax.set_title(dataset.CLASS_NAMES[ax_y[i]]);ax.axis("off")
fig.tight_layout();fig.savefig(CURVES/"augmentation_check.png");plt.close(fig)
del net,opt,x,y;torch.cuda.empty_cache()

def config(exp_id,backbone,seed=0,**params):
    return train.Config(exp_id=exp_id,backbone=backbone,seed=seed,epochs=EPOCHS,batch_size=BATCH,
        images_dir=str(images_dir),labels_dir=str(labels),out_dir=str(RUNS),pred_dir=str(PRED),
        curves_dir=str(CURVES),num_workers=2,**params)

def execute(cfg):
    print("START",cfg.exp_id,cfg.seed,cfg.backbone,flush=True)
    result=train.run(cfg)
    save_json(train.run_dir(cfg)/"result.json",result)
    return result

def metrics(cfg,split="val"):
    p=ev.read_pred(train.pred_path(cfg,split))
    return ev.compute_metrics(p.y_true,p.y_pred,p.probs)

def checkpoint_net(cfg):
    net=model.build_model(cfg.backbone,False,init="scratch").cuda().eval()
    net.load_state_dict(torch.load(train.run_dir(cfg)/"best_model.pt",map_location="cuda",weights_only=True))
    return net

backbones=["resnet50","convnext_tiny","swin_tiny_patch4_window7_224","efficientnet_b0","mobilenetv3_large_100"]
for i,name in enumerate(backbones,1):
    cfg=config(f"B{i:02d}",name);r=execute(cfg);m=metrics(cfg);net=checkpoint_net(cfg)
    lat=benchmark.latency_report(net,device="cuda",batch_size=1,warmup=10,iters=100)
    tag=timm.models.get_pretrained_cfg(name).tag
    TABLES["Backbones"].append({**r,"tag":tag,"resolution":224,"epochs":EPOCHS,
        "val_macro_f1":m["macro_f1"],"val_top1":m["top1"],"latency_p95_ms":lat["p95"]})
    TABLES["Latency"].append({"exp_id":cfg.exp_id,**lat,"fused_bn":False})
    del net;torch.cuda.empty_cache();persist()
selected=max(TABLES["Backbones"],key=lambda r:r["val_macro_f1"])["backbone"]
recipes=[("T00","Baseline",{}),("T01","Initialization",{"init":"frozen"}),
    ("T02","Initialization",{"init":"scratch"}),("T03","Augmentation",{"aug":"color"}),
    ("T04","Augmentation",{"mix":"cutmix"}),("T05","Loss",{"loss":"focal"}),
    ("T06","Loss",{"loss":"ce_weighted"})]
recipe_cfg={}
base_m=None
for exp,axis,params in recipes:
    cfg=config(exp,selected,**params);execute(cfg);m=metrics(cfg);recipe_cfg[exp]=params
    if exp=="T00":base_m=m
    TABLES["Training"].append({"exp_id":exp,"backbone":selected,"axis":axis,"changes":json.dumps(params),
        "seed":0,"val_macro_f1":m["macro_f1"],"val_top1":m["top1"],
        "delta_macro_f1":m["macro_f1"]-base_m["macro_f1"],"chinee_apple_f1":m["f1"][0],"snake_weed_f1":m["f1"][7]})
    persist()
# Combine the validation winner of each independent axis (baseline permitted).
combined={}
for axis in ["Initialization","Augmentation","Loss"]:
    candidates=[r for r in TABLES["Training"] if r["axis"] in ("Baseline",axis)]
    winner=max(candidates,key=lambda r:r["val_macro_f1"])
    combined.update(recipe_cfg[winner["exp_id"]])
recipe_cfg["T07"]=combined
cfg=config("T07",selected,**combined);execute(cfg);m=metrics(cfg)
TABLES["Training"].append({"exp_id":"T07","backbone":selected,"axis":"Combination","changes":json.dumps(combined),
    "seed":0,"val_macro_f1":m["macro_f1"],"val_top1":m["top1"],"delta_macro_f1":m["macro_f1"]-base_m["macro_f1"],
    "chinee_apple_f1":m["f1"][0],"snake_weed_f1":m["f1"][7]});persist()
best_recipe=max(TABLES["Training"],key=lambda r:r["val_macro_f1"])["exp_id"]
params=recipe_cfg[best_recipe]
cfg=config(best_recipe,selected,**params);net=checkpoint_net(cfg)
vloader=dataset.make_loader(va,images_dir,dataset.build_transforms(False),BATCH,False,num_workers=2)
names,y,z=inference.predict_logits(net,vloader,"cuda")
_,_,flip=inference.predict_logits(net,vloader,"cuda",inference.view_hflip)
T=inference.fit_temperature(z,y)
fused=inference.fuse_conv_bn(net)
_,_,zf=inference.predict_logits(fused,vloader,"cuda")
np.testing.assert_allclose(zf,z,rtol=1e-4,atol=1e-4)
probabilities={"I00":inference.apply_temperature(z,1),"I01":inference.aggregate_views([z,flip],"prob"),
    "I03":inference.aggregate_views([z,flip],"logit"),"I07":inference.apply_temperature(z,T),"I08":inference.apply_temperature(zf,1)}
base_lat=None
for method,probs in probabilities.items():
    m=ev.compute_metrics(y,probs.argmax(1),probs)
    if method in ("I01","I03"):
        lat=benchmark.tta_latency(net,2,space="prob" if method=="I01" else "logit",device="cuda",batch_size=1,warmup=10,iters=100)
    else:
        lat=benchmark.latency_report(fused if method=="I08" else net,device="cuda",batch_size=1,warmup=10,iters=100)
    if method=="I00":base_lat=lat
    TABLES["Inference"].append({"exp_id":method,"checkpoint":best_recipe,"method":method,"K":2 if method in ("I01","I03") else 1,
        "val_macro_f1":m["macro_f1"],"val_top1":m["top1"],"ece":m["ece"],"temperature":T if method=="I07" else 1,
        "p50":lat["p50"],"p95":lat["p95"],"p99":lat["p99"],"images_per_s":lat["images_per_s"],"relative_cost":lat["p50"]/base_lat["p50"]})
    TABLES["Latency"].append({"exp_id":method,**lat,"fused_bn":method=="I08"})
    ev.save_predictions(PRED/f"{method}_seed0_val.csv",names,y,probs)
    persist()
# Also measure batch-32 throughput for the final model.
TABLES["Latency"].append({"exp_id":"I00_batch32",**benchmark.latency_report(net,device="cuda",batch_size=32,warmup=10,iters=100),"fused_bn":False})
winner=sorted(TABLES["Inference"],key=lambda r:(-r["val_macro_f1"],r["ece"],r["p95"]))[0]
locked={"backbone":selected,"recipe_id":best_recipe,"recipe":params,"inference_method":winner["exp_id"],
        "selection":"val macro-F1 descending; ECE ascending; p95 ascending", "seeds":[0,1,2]}
save_json(SUB/"selection_locked_before_test.json",locked)
print("LOCKED_BEFORE_TEST",locked,flush=True)
del net,fused;torch.cuda.empty_cache()
for exp,recipe,method in [("T00",{},"I00"),("F01",params,winner["exp_id"])]:
    for seed in [0,1,2]:
        cfg=config(exp,selected,seed=seed,save_test_predictions=True,inference_method=method,**recipe)
        r=execute(cfg);vm=metrics(cfg);tm=metrics(cfg,"test")
        TABLES["Final"].append({"exp_id":exp,"backbone":selected,"recipe":json.dumps(recipe),"inference":method,"seed":seed,
            "val_macro_f1":vm["macro_f1"],"test_macro_f1":tm["macro_f1"],"test_top1":tm["top1"],"test_ece":tm["ece"]})
        for c,name in enumerate(dataset.CLASS_NAMES):
            TABLES["PerClass"].append({"exp_id":exp,"seed":seed,"class":name,"n_test":int((te.Label==c).sum()),
                "precision":tm["precision"][c],"recall":tm["recall"][c],"f1":tm["f1"][c]})
        persist()
for exp in ["T00","F01"]:
    group=ev.load_group(str(PRED/f"{exp}_seed*_test.csv"),str(labels/"test_subset0.csv"))
    ev.save_group(SUB/"eval_out",exp,group,dataset.CLASS_NAMES)
    text=ev.report_group(exp,group,dataset.CLASS_NAMES)
    (SUB/f"{exp}_evaluation.txt").write_text(text,encoding="utf-8")
    print(text,flush=True)
    rows=[r for r in TABLES["Final"] if r["exp_id"]==exp]
    summary={"exp_id":exp+"_mean_std","seed":"mean +/- sample std"}
    for key in ["val_macro_f1","test_macro_f1","test_top1","test_ece"]:
        values=[r[key] for r in rows];summary[key]=float(np.mean(values));summary[key+"_std"]=float(np.std(values,ddof=1))
    TABLES["Final"].append(summary)
TABLES["Summary"]=sorted(TABLES["Backbones"]+TABLES["Training"],key=lambda r:r["val_macro_f1"],reverse=True)[:10]
persist()
# Export confusion matrix and genuine misclassification examples for final seed 0.
p=ev.read_pred(PRED/"F01_seed0_test.csv")
cm=ev.confusion_matrix(p.y_true,p.y_pred)
fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(cm,cmap="Blues");fig.colorbar(im,ax=ax)
ax.set_xticks(range(9),dataset.CLASS_NAMES,rotation=45,ha="right");ax.set_yticks(range(9),dataset.CLASS_NAMES)
ax.set_xlabel("Predicted");ax.set_ylabel("True")
for i in range(9):
    for j in range(9):ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8)
fig.tight_layout();fig.savefig(CURVES/"F01_seed0_confusion.png");plt.close(fig)
errors=np.flatnonzero(p.y_true!=p.y_pred)[:12]
if len(errors):
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for ax,i in zip(axes.flat,errors):
        with Image.open(images_dir/str(p.filenames[i])) as image:ax.imshow(image)
        ax.set_title(f"true={dataset.CLASS_NAMES[p.y_true[i]]}\npred={dataset.CLASS_NAMES[p.y_pred[i]]}",fontsize=8);ax.axis("off")
    for ax in list(axes.flat)[len(errors):]:ax.axis("off")
    fig.tight_layout();fig.savefig(CURVES/"F01_seed0_errors.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for row in TABLES["Inference"]:ax.scatter(row["p95"],row["val_macro_f1"]);ax.annotate(row["exp_id"],(row["p95"],row["val_macro_f1"]))
ax.set_xlabel("p95 latency (ms, batch 1)");ax.set_ylabel("Validation macro-F1");fig.tight_layout();fig.savefig(CURVES/"inference_tradeoff.png");plt.close(fig)
RUNTIME["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());RUNTIME["status"]="complete"
save_json(SUB/"runtime.json",RUNTIME)
# Bundle small reproducibility artifacts; checkpoints remain in private Kaggle output.
import zipfile
with zipfile.ZipFile(ROOT/"submission_bundle.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for path in SUB.rglob("*"):
        if path.is_file() and path.suffix not in (".pt",".pyc"):
            archive.write(path,path.relative_to(ROOT))
print("COMPLETE",RUNTIME,flush=True)
