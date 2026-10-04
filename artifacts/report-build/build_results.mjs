import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const sub = path.resolve(process.argv[2] || 'submissions/2A202602873_thai_phuc_tien');
const tables = JSON.parse(await fs.readFile(path.join(sub, 'tables.json'), 'utf8'));
const workbook = Workbook.create();
const fields = {
  Summary: ['exp_id','backbone','val_macro_f1','val_top1','latency_p95_ms','changes'],
  Final: ['exp_id','seed','backbone','recipe','inference','val_macro_f1','val_macro_f1_std','test_macro_f1','test_macro_f1_std','test_top1','test_top1_std','test_ece','test_ece_std'],
  Backbones: ['exp_id','backbone','tag','params_m','gmacs','resolution','epochs','seed','val_macro_f1','val_top1','best_epoch','time_per_epoch_s','latency_p95_ms'],
  Training: ['exp_id','backbone','axis','changes','seed','val_macro_f1','val_top1','delta_macro_f1','chinee_apple_f1','snake_weed_f1'],
  Inference: ['exp_id','method','checkpoint','K','val_macro_f1','val_top1','ece','temperature','p50','p95','p99','images_per_s','relative_cost','note'],
  PerClass: ['exp_id','seed','class','n_test','precision','precision_std','recall','recall_std','f1','f1_std'],
  Latency: ['exp_id','gpu','dtype','batch','img_size','fused_bn','p50','p95','p99','images_per_s','torch'],
};
const labels = {params_m:'Parameters (M)',gmacs:'GMAC',resolution:'Input (px)',time_per_epoch_s:'Train/epoch (s)',latency_p95_ms:'p95 (ms)',n_test:'Test images/seed',p50:'p50 (ms)',p95:'p95 (ms)',p99:'p99 (ms)',images_per_s:'Images/s',K:'Views (K)',delta_macro_f1:'Delta val macro-F1',fused_bn:'BN fusion applied'};
const label = k => labels[k] || k.replaceAll('_',' ');
const column = n => { let s=''; for(n++;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s; };

for (const [name,keys] of Object.entries(fields)) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const rows = tables[name];
  if (!rows?.length) throw new Error(`Missing ${name} data`);
  const end = column(keys.length-1);
  sheet.getRange('A2').values = [[`DeepWeeds — ${name}`]];
  sheet.getRange('A2').format.font = {name:'Arial',size:14,bold:true,color:'#172554'};
  sheet.getRange(`A4:${end}4`).values = [keys.map(label)];
  sheet.getRange(`A5:${end}${4+rows.length}`).values = rows.map(row => keys.map(k => {
    const value = k==='seed' && typeof row[k]==='string' ? 'mean ± std' : row[k];
    if (value === null || value === undefined) return 'n.a.';
    return typeof value === 'object' ? JSON.stringify(value) : value;
  }));
  const range = sheet.getRange(`A4:${end}${4+rows.length}`);
  range.format.font = {name:'Arial',size:10,color:'#172033'};
  range.format.rowHeight = 24;
  range.format.verticalAlignment = 'center';
  sheet.getRange(`A4:${end}4`).format = {fill:'#243B53',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:42,wrapText:true,horizontalAlignment:'center'};
  keys.forEach((key,index) => {
    const col=column(index), body=sheet.getRange(`${col}5:${col}${4+rows.length}`);
    const width = ['backbone','method','recipe','changes','note'].includes(key) ? 36 : Math.max(15,Math.min(23,label(key).length+2));
    sheet.getRange(`${col}4:${col}${4+rows.length}`).format.columnWidth = width;
    body.format.horizontalAlignment = rows.some(row=>typeof row[key]==='number') ? 'right' : 'left';
    if (['recipe','changes','method','note','backbone'].includes(key)) {body.format.wrapText=true;body.format.rowHeight=42;}
    if (/f1|ece|top1|precision|recall/.test(key)) body.format.numberFormat = key.includes('top1') ? '0.00%' : '0.0000';
    else if (/p50|p95|p99|time_per_epoch|params_m|gmacs|images_per_s|relative_cost|temperature/.test(key)) body.format.numberFormat='0.000';
    else if (['epochs','seed','resolution','best_epoch','K','n_test','batch','img_size'].includes(key)) body.format.numberFormat='0';
  });
  if (rows.length>12 || name!=='Summary') sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(1);
  if (name==='Final') {
    // The measured per-seed records remain unchanged; summary calculations link to them.
    for (const [index,first,last] of [[6,5,7],[7,8,10]]) {
      const summary=5+index;
      sheet.getRange(`C${summary}:E${summary}`).values=[['convnext_tiny',index===6?'{}':'{"loss":"focal"}',index===6?'I00':'I01']];
      for(const key of ['val_macro_f1','test_macro_f1','test_top1','test_ece']) {
        const col=column(keys.indexOf(key)), stdcol=column(keys.indexOf(key+'_std'));
        sheet.getRange(`${col}${summary}`).formulas=[[`=AVERAGE(${col}${first}:${col}${last})`]];
        sheet.getRange(`${stdcol}${summary}`).formulas=[[`=STDEV.S(${col}${first}:${col}${last})`]];
      }
      sheet.getRange(`A${summary}:${end}${summary}`).format.fill='#E8EEF6';
    }
  }
  if (name==='Inference') {
    sheet.getRange(`A5:${end}${4+rows.length}`).format.rowHeight=72;
    const noop=rows.findIndex(r=>r.exp_id==='I08');
    if(noop>=0) sheet.getRange(`A${noop+5}:${end}${noop+5}`).format.fill='#FFF2CC';
  }
  if(name==='Summary') {
    sheet.getRange('A16').values=[['Final test comparison: three seeds (sample standard deviation)']];
    sheet.getRange('A17:F17').values=[['Configuration','Macro-F1 mean','Macro-F1 std','Top-1 mean','Top-1 std','ECE mean']];
    sheet.getRange('A17:F17').format={fill:'#243B53',font:{color:'#FFFFFF',bold:true},rowHeight:32,wrapText:true};
    for(const [dest,src] of [[18,11],[19,12]]) {
      sheet.getRange(`B${dest}:F${dest}`).format.numberFormat='0.0000';
      sheet.getRange(`D${dest}:E${dest}`).format.numberFormat='0.00%';
    }
    const n=22;
    sheet.getRange(`A${n}`).values=[['Final selection: ConvNeXt-Tiny + focal loss + horizontal-flip TTA.']];
    sheet.getRange(`A${n+1}`).values=[['Test macro-F1 does not exceed the baseline beyond seed variation.']];
    sheet.getRange(`A${n+2}`).values=[['Source: Kaggle GPU logs and predictions; all metrics recomputed by eval.py.']];
    sheet.tabColor='#243B53';
  }
}
for(const [dest,src] of [[18,11],[19,12]]) workbook.worksheets.getItem('Summary').getRange(`A${dest}:F${dest}`).formulas=[[`=Final!A${src}`,`=Final!H${src}`,`=Final!I${src}`,`=Final!J${src}`,`=Final!K${src}`,`=Final!L${src}`]];
workbook.recalculate();
await fs.mkdir(path.join(sub,'workbook_previews'),{recursive:true});
for(const name of Object.keys(fields)) {
  const end=column(fields[name].length-1), last=name==='Summary'?24:Math.min(4+tables[name].length,14);
  const preview=await workbook.render({sheetName:name,range:`A1:${end}${last}`,scale:1,format:'png'});
  await fs.writeFile(path.join(sub,'workbook_previews',`${name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const check=await workbook.inspect({kind:'table',range:'Final!F11:M12',include:'values,formulas',maxChars:2200,tableMaxRows:2,tableMaxCols:8});
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!',options:{useRegex:true,maxResults:30},maxChars:1500});
await fs.writeFile(path.join(sub,'workbook_checks.txt'),check.ndjson+'\n'+errors.ndjson);
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(sub,'results.xlsx'));
console.log('Saved results.xlsx and seven sheet previews.');
