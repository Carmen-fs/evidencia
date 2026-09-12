from pathlib import Path
import subprocess, json
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Reutiliza únicamente las funciones de formato de la evidencia anterior.
helper=Path(__file__).resolve().parents[1]/'evidencia_trabajo/crear_documento.py'
prefix=helper.read_text(encoding='utf-8').split("ptext('Evidencia de extracción y análisis de datos','Title')")[0]
exec(compile(prefix, str(helper), 'exec'))
doc.core_properties.title='Evidencia de Leads de Piso 2025 y BCS'
doc.core_properties.subject='Extracción, limpieza, características y regresión lineal'
doc.styles['Normal'].paragraph_format.line_spacing=1.05
POPPLER=Path(r'C:\Users\maica\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
pdfmetrics.registerFont(TTFont('ArialLocal',r'C:\Windows\Fonts\arial.ttf'))
FONT='ArialLocal'
BLUE=colors.HexColor('#327a96'); RED=colors.HexColor('#c86359')
src_leads=ROOT/'Valores_nulos_2/Valores-Nulos/Actividad 2.2/leads_piso_2025_sin_nulos.csv'
src_bcs=ROOT/'Regresion lineal simple/Regresion-lineal-simple/Actividad 5.2/Analisis GAC Full7agosto 2026( BCS).csv'
raw_leads=pd.read_csv(src_leads,dtype=str)
leads=raw_leads.copy()
leads.columns=leads.columns.str.strip()
leads['Asesor']=leads['Asesor'].str.replace('\u00a0',' ',regex=False).str.replace(r'\s+',' ',regex=True).str.strip().str.upper()
months=['ene','feb','mar','abr']
leads[months]=leads[months].apply(pd.to_numeric,errors='raise')
assert leads[months].notna().all().all()
assert (leads[months]>=0).all().all()
admin_names=['CASA','DEMO VENDEDOR','USUARIO DE BDC Y CASA','SYS_COMPETITION']
admin=leads[leads.Asesor.isin(admin_names)].copy()
leads=leads[~leads.Asesor.isin(admin_names)].groupby('Asesor',as_index=False)[months].sum()
leads['Total']=leads[months].sum(axis=1)
totals=leads[months].sum()
raw_bcs=pd.read_csv(src_bcs,header=None,dtype=str)
bcs=raw_bcs.iloc[:,[1,3,4]].copy()
bcs.columns=['Modelo','Objetivo','Ventas_Reales']
bcs.Modelo=bcs.Modelo.str.strip().str.upper()
bcs=bcs[bcs.Modelo.isin(['EMZOOM','AION','EMKOO','GS4','GN8','GS8'])].reset_index(drop=True)
for col in ['Objetivo','Ventas_Reales']:
    bcs[col]=pd.to_numeric(bcs[col].str.replace(',','',regex=False).str.strip(),errors='raise')
bcs['Brecha']=bcs.Ventas_Reales-bcs.Objetivo
bcs['Cumplimiento']=bcs.Ventas_Reales/bcs.Objetivo*100
def ols(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    slope,intercept=np.polyfit(x,y,1)
    pred=intercept+slope*x
    return {'slope':float(slope),'intercept':float(intercept),'r':float(np.corrcoef(x,y)[0,1]),
            'r2':float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()),'pred':pred,'resid':y-pred}
lm=ols(range(1,5),totals)
bm=ols(bcs.Objetivo,bcs.Ventas_Reales)
assert int(totals.sum())==244 and len(leads)==35
assert int(raw_leads[months].astype(int).to_numpy().sum())==int(totals.sum()+admin[months].to_numpy().sum())
assert np.isclose(bm['r2'],.47456385131536927)
assert np.isclose(lm['r2'],lm['r']**2)
def iqr(s):
    q1,q3=s.quantile([.25,.75]); spread=q3-q1
    lo=q1-1.5*spread; hi=q3+1.5*spread
    return [q1,q3,lo,hi,int(((s<lo)|(s>hi)).sum())]
stats={'leads_months':totals.to_dict(),'leads_total':int(totals.sum()),'leads_n':len(leads),
       'iqr_total':iqr(leads.Total),'iqr_monthly':iqr(totals),'iqr_obj':iqr(bcs.Objetivo),'iqr_sales':iqr(bcs.Ventas_Reales),
       'models':{k:{kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in v.items()} for k,v in [('leads',lm),('bcs',bm)]}}
(OUT/'estadisticas.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')

def save_plot(d,name):
    pdf=OUT/(name+'.pdf'); renderPDF.drawToFile(d,str(pdf))
    subprocess.run([str(POPPLER),'-singlefile','-scale-to','1600','-png',str(pdf),str(OUT/name)],check=True,capture_output=True)
    return OUT/(name+'.png')
def title(d,t): d.add(String(310,272,t,textAnchor='middle',fontName=FONT,fontSize=13))
def bars(name,labels,data,title_text,legend=None):
    d=Drawing(620,290); title(d,title_text)
    bc=VerticalBarChart();bc.x=55;bc.y=48;bc.width=525;bc.height=195;bc.data=data
    bc.categoryAxis.categoryNames=labels
    bc.categoryAxis.labels.fontName=FONT;bc.categoryAxis.labels.fontSize=10
    bc.valueAxis.labels.fontName=FONT;bc.valueAxis.labels.fontSize=9
    bc.valueAxis.valueMin=0;bc.valueAxis.valueMax=max(max(a) for a in data)*1.18
    bc.bars[0].fillColor=BLUE
    if len(data)>1:bc.bars[1].fillColor=RED
    bc.bars.strokeColor=None
    bc.barLabels.fontName=FONT;bc.barLabels.fontSize=9;bc.barLabelFormat='%d'
    d.add(bc)
    if legend:
        for i,l in enumerate(legend):
            d.add(Rect(185+i*135,12,10,8,fillColor=[BLUE,RED][i],strokeColor=None))
            d.add(String(200+i*135,12,l,fontName=FONT,fontSize=10))
    return save_plot(d,name)
def regression(name,x,y,model,ttl,xlabel,ylabel,labels=None):
    d=Drawing(620,290);title(d,ttl)
    plot=LinePlot();plot.x=58;plot.y=52;plot.width=500;plot.height=190
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    plot.data=[list(zip(x,y)),[(float(x.min()),float(model['intercept']+model['slope']*x.min())),(float(x.max()),float(model['intercept']+model['slope']*x.max()))]]
    plot.lines[0].strokeColor=None
    marker=makeMarker('FilledCircle');marker.size=7;marker.fillColor=BLUE;marker.strokeColor=BLUE
    plot.lines[0].symbol=marker;plot.lines[1].strokeColor=RED;plot.lines[1].strokeWidth=1.7
    plot.xValueAxis.valueMin=0;plot.xValueAxis.valueMax=float(x.max()*1.14)
    plot.yValueAxis.valueMin=0;plot.yValueAxis.valueMax=float(y.max()*1.2)
    for a in [plot.xValueAxis,plot.yValueAxis]:a.labels.fontName=FONT;a.labels.fontSize=9
    if x.max()==4:plot.xValueAxis.valueSteps=[1,2,3,4]
    d.add(plot)
    d.add(String(310,25,xlabel,textAnchor='middle',fontName=FONT,fontSize=10))
    d.add(String(60,250,ylabel,fontName=FONT,fontSize=10))
    d.add(String(408,250,'Puntos: observado   Línea: ajuste',fontName=FONT,fontSize=8))
    if labels:
        for i,(xx,yy,l) in enumerate(zip(x,y,labels)):
            dx=5;dy=7
            if l=='AION': dx=-12;dy=12
            if l=='GS4':dx=6;dy=-12
            d.add(String(58+500*xx/plot.xValueAxis.valueMax+dx,52+190*yy/plot.yValueAxis.valueMax+dy,l,fontName=FONT,fontSize=8))
    return save_plot(d,name)
def heatmaps():
    d=Drawing(620,275)
    specs=[(leads[months].corr().values,['Ene','Feb','Mar','Abr'],40,35,48,'Leads por asesor   n = 35'),
           (bcs[['Objetivo','Ventas_Reales']].corr().values,['Objetivo','Ventas'],365,75,67,'BCS   n = 6')]
    for a,labels,x0,y0,size,ttl in specs:
        n=len(labels);d.add(String(x0+n*size/2,y0+n*size+35,ttl,textAnchor='middle',fontName=FONT,fontSize=12))
        for i in range(n):
            d.add(String(x0-7,y0+(n-i-.5)*size-3,labels[i],textAnchor='end',fontName=FONT,fontSize=9))
            d.add(String(x0+(i+.5)*size,y0-16,labels[i],textAnchor='middle',fontName=FONT,fontSize=9))
            for j in range(n):
                v=a[i,j];c=colors.Color(1-.80*max(v,0),1-.55*abs(v),1-.4*max(v,0))
                d.add(Rect(x0+j*size,y0+(n-i-1)*size,size,size,fillColor=c,strokeColor=colors.white))
                d.add(String(x0+(j+.5)*size,y0+(n-i-.5)*size-4,f'{v:.2f}',textAnchor='middle',fontName=FONT,fontSize=11))
    return save_plot(d,'correlaciones')
def figure(path,caption,width=6.3,maxheight=2.85):
    with Image.open(path) as im: width=min(width,maxheight*im.width/im.height)
    p=doc.add_paragraph();p.alignment=1;p.paragraph_format.space_after=Pt(2)
    p.add_run().add_picture(str(path),width=Inches(width))
    ptext(caption,'Caption')

monthly_fig=bars('leads_mensuales',['Enero','Febrero','Marzo','Abril'],[list(totals)],'Leads de Piso 2025 por mes')
bcs_fig=bars('bcs_modelos',list(bcs.Modelo),[list(bcs.Objetivo),list(bcs.Ventas_Reales)],'Objetivo y ventas reales por modelo',['Objetivo','Ventas reales'])
reg_leads=regression('reg_leads',range(1,5),totals,lm,'Regresión del mes y los leads de piso','Mes de 2025','Leads')
reg_bcs=regression('reg_bcs',bcs.Objetivo,bcs.Ventas_Reales,bm,'Regresión del objetivo y las ventas reales','Objetivo de ventas','Ventas reales',list(bcs.Modelo))
corr_fig=heatmaps()

ptext('Evidencia de extracción y análisis de datos','Title')
ptext('Leads de Piso 2025 y BCS','Subtitle')
ptext('En esta evidencia analizo dos bases del socio formador: Leads de Piso 2025, que contiene registros mensuales por asesor, y BCS, de la cual selecciono los objetivos y las ventas reales por modelo de vehículo. El propósito es preparar la información, describir su distribución y examinar relaciones mediante regresión lineal simple.')
ptext('El análisis de Leads de Piso reúne 244 registros de enero a abril entre 35 asesores, después de separar cuentas administrativas y agrupar un nombre duplicado. En BCS, los seis modelos seleccionados suman 41 ventas reales frente a 44 unidades de objetivo por modelo. La concentración de registros y las diferencias entre metas requieren interpretarse junto con la calidad de captura.')
h('Bases principales')
table(['Base','Archivo de entrada','Unidad de análisis'],[
('Leads de Piso 2025','leads_piso_2025_sin_nulos.csv','Asesor con conteos de enero a abril de 2025'),
('BCS','Analisis GAC Full7agosto 2026( BCS).csv','Modelo de vehículo en la sección de ventas')],[1.35,3.45,2.1])
h('Etapas del trabajo')
for t in ['Extracción y transformación de las dos bases.','Verificación de nulos y ejemplo complementario de sustitución.','Detección de valores atípicos y justificación de su conservación.','Extracción de características y elaboración de gráficas.','Regresión lineal simple y análisis de correlación.','Hallazgos, propuestas y conclusión.']:
    ptext(t,'List Bullet')
h('Alcance')
ptext('Leads de Piso contiene cantidades registradas, no identificadores de clientes únicos ni ventas. BCS no es una serie mensual de 2025: su encabezado incluye la fecha 24 de agosto de 2026. Las bases se analizan por separado porque no cuentan con una clave y un periodo común que justifiquen unirlas.')
ptext('Citas Digitales se utiliza únicamente como ejemplo complementario de los métodos de nulos y atípicos trabajados en las actividades 2.1 y 3.1. Los resultados principales corresponden a Leads de Piso 2025 y BCS.')

page('Extracción y transformación de Leads de Piso')
h('Lectura y normalización')
ptext('El archivo de entrada contiene 40 filas y cinco columnas: Asesor, ene, feb, mar y abr. Está guardado como una versión sin nulos. Limpio espacios en los encabezados y normalizo el campo Asesor con mayúsculas y un solo espacio entre palabras. Después convierto los cuatro meses a conteos numéricos.')
code(r'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

leads = pd.read_csv("leads_piso_2025_sin_nulos.csv", dtype=str)
leads.columns = leads.columns.str.strip()
leads["Asesor"] = (
    leads["Asesor"].str.replace("\u00a0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True).str.strip().str.upper()
)
meses = ["ene", "feb", "mar", "abr"]
leads[meses] = leads[meses].apply(pd.to_numeric, errors="raise")''')
h('Separación de cuentas administrativas')
ptext('Aplico el criterio de separación de cuentas administrativas utilizado en la preparación de ventas de la actividad 5.2. CASA, DEMO VENDEDOR, USUARIO DE BDC Y CASA y SYS_COMPETITION quedan fuera del análisis por asesor. CASA aporta dos registros de febrero y uno de marzo; las otras tres cuentas tienen cero en los cuatro meses.')
code('''administrativas = ["CASA", "DEMO VENDEDOR",
                   "USUARIO DE BDC Y CASA", "SYS_COMPETITION"]
leads = leads[~leads["Asesor"].isin(administrativas)].copy()
leads = leads.groupby("Asesor", as_index=False)[meses].sum()
leads["Total"] = leads[meses].sum(axis=1)''')
ptext('Al normalizar espacios, dos filas de JOSE MANUEL LOPEZ coinciden; ambas contienen cero y se agrupan en una sola. No fusiono nombres diferentes como PAUL ROSAS y PAUL ROSAS IRIGOYIN sin un identificador que confirme que se trata de la misma persona.')
table(['Revisión','Entrada','Resultado'],[
('Filas o asesores','40 filas','35 asesores normalizados'),('Suma de los cuatro meses','247 registros','244 registros'),('Cuentas administrativas','4 filas','3 registros separados'),('Nombres repetidos tras normalizar','2 filas de un mismo nombre','1 fila agrupada')],[2.85,1.7,2.35])

page('Extracción y transformación de BCS')
h('Selección de la sección de ventas')
ptext('BCS contiene un tablero con secciones de ventas, postventa y otros indicadores. La lectura sin encabezado tiene 53 filas y 56 columnas. Para este análisis selecciono las columnas que contienen el nombre del indicador, el objetivo y el resultado real, y conservo únicamente los seis modelos trabajados en la actividad 5.2.')
code('''bcs_raw = pd.read_csv(
    "Analisis GAC Full7agosto 2026( BCS).csv",
    header=None, dtype=str
)
bcs = bcs_raw.iloc[:, [1, 3, 4]].copy()
bcs.columns = ["Modelo", "Objetivo", "Ventas_Reales"]
bcs["Modelo"] = bcs["Modelo"].str.strip().str.upper()
modelos = ["EMZOOM", "AION", "EMKOO", "GS4", "GN8", "GS8"]
bcs = bcs[bcs["Modelo"].isin(modelos)].reset_index(drop=True)
for columna in ["Objetivo", "Ventas_Reales"]:
    bcs[columna] = pd.to_numeric(
        bcs[columna].str.replace(",", "", regex=False).str.strip(),
        errors="raise"
    )''')
table(['Modelo','Objetivo','Ventas reales'],[(r.Modelo,int(r.Objetivo),int(r.Ventas_Reales)) for r in bcs.itertuples()],[2.9,2,2])
h('Comprobación de consistencia')
ptext('La base seleccionada contiene seis filas y tres columnas, sin nulos ni modelos repetidos. Se preserva el valor cero de GN8 porque es un resultado registrado, no una celda vacía.')
ptext('Las metas de los seis modelos suman 44, pero la fila general de número de ventas del tablero muestra un objetivo de 40. Las ventas reales suman 41 en ambos casos. Por tanto, 41/44 = 93.18% describe el cumplimiento de las metas desglosadas por modelo; 41/40 = 102.50% utiliza el objetivo general. No son el mismo indicador y debe aclararse la discrepancia antes de usarlo en decisiones.')
ptext('El nombre del archivo menciona 7 de agosto de 2026, mientras que el encabezado interno incluye 24 de agosto de 2026. No se asigna a estos valores un periodo mensual específico ni se mezclan con los conteos de Leads de Piso de 2025.')

page('Limpieza de nulos')
h('Verificación en las bases principales')
ptext('En Leads de Piso 2025, el archivo disponible ya tiene cero nulos en sus 40 filas y cinco columnas. Después de la preparación, Asesor y los cuatro meses siguen sin nulos. En BCS, las columnas seleccionadas de los seis modelos tampoco tienen nulos. No se requiere imputación adicional en estas variables.')
code('''print(leads[["Asesor"] + meses].isna().sum())
print(bcs.isna().sum())
assert leads[meses].notna().all().all()
assert (leads[meses] >= 0).all().all()
assert bcs[["Objetivo", "Ventas_Reales"]].notna().all().all()''')
ptext('Los espacios vacíos del tablero completo de BCS separan secciones y encabezados. No se rellenan indiscriminadamente: primero se extrae la tabla de interés. Del mismo modo, la ausencia de nulos en el CSV de leads no permite reconstruir cuántos vacíos existían antes de su limpieza ni cómo se sustituyeron.')
h('Interpretación de los ceros en Leads de Piso')
ptext('Veinte de los 35 asesores presentan cero registros en los cuatro meses, equivalentes al 57.14%. Estos valores se conservan como están en el archivo. Sin información de altas, bajas o cobertura de captura, no es posible asegurar que todos signifiquen ausencia real de atención a clientes.')
h('Ejemplo complementario de Citas Digitales')
ptext('En la actividad 2.1, Citas_Digital_Filtrado.csv tenía 676 filas y 16 columnas, con 3,557 celdas nulas. Allí se utilizaron textos para información ausente, moda para ciertas categorías e indicadores y mediana para Potencial de compra. La verificación final reportó cero nulos.')
code('''# Ejemplo de la actividad 2.1, independiente de leads y BCS
citas["Fecha"] = citas["Fecha"].fillna("Sin fecha")
citas["Potencial de compra"] = pd.to_numeric(
    citas["Potencial de compra"], errors="coerce")
citas["Potencial de compra"] = citas["Potencial de compra"].fillna(
    citas["Potencial de compra"].median())''')
ptext('Este fragmento supone la corrección previa de formatos como 80%% y 80& a 0.8, realizada en esa actividad. Sustituir un nulo permite procesar la tabla, pero no recupera la fecha, el estatus o el valor real perdido; por eso conviene marcar las celdas imputadas.')

page('Valores atípicos')
h('Detección con rango intercuartílico')
ptext('Calculo Q1, Q3 e IQR = Q3 − Q1. Los límites de detección son Q1 − 1.5 IQR y Q3 + 1.5 IQR. Aplico el método a los totales de leads por asesor y, por separado, a los objetivos y ventas de BCS.')
table(['Variable','Q1','Q3','Límite inferior','Límite superior','Casos'],[
('Total de leads por asesor','0','8.5','−12.75','21.25','4'),
('Objetivo BCS','2.25','8.5','−7.125','17.875','1'),
('Ventas reales BCS','3.5','11.75','−8.875','24.125','0')],[2.1,.55,.55,1.25,1.25,1.2])
ptext('Los totales de Alan Gonzales (51), Pedro Gonzales (44), Aurelio Felipe (26) y Mauricio Alfredo (22) superan el límite de leads. El objetivo de EMZOOM (22) supera el límite de objetivos de BCS. Son observaciones estadísticamente extremas; no hay evidencia suficiente para declararlas errores.')
code('''def detectar_atipicos(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)

leads["Atipico_Total"] = detectar_atipicos(leads["Total"])
bcs["Atipico_Objetivo"] = detectar_atipicos(bcs["Objetivo"])''')
h('Justificación del tratamiento')
ptext('Conservo los casos detectados. El volumen de leads puede diferir por asignaciones, antigüedad o hábitos de captura; sustituir a los asesores con más registros por la mediana reduciría artificialmente el total. En BCS, los objetivos varían por modelo y no se espera que sean idénticos. Además, seis modelos constituyen una muestra pequeña para establecer límites estables.')
h('Ejemplo complementario de sustitución')
ptext('En Citas Digitales, actividad 3.1, el IQR de Potencial de compra fue 0.10 y los límites 0.55 y 0.95. Se detectaron 71 de 676 valores (10.50%) y se sustituyeron por 0.75, la mediana de los valores restantes. Este tratamiento pertenece a esa base y no se aplica a los conteos de leads ni a los objetivos de BCS.')
code('''# Ejemplo independiente de la actividad 3.1
mascara = detectar_atipicos(citas["Potencial de compra"])
serie = citas["Potencial de compra"].mask(mascara)
citas["Potencial de compra"] = serie.fillna(serie.median())''')
ptext('Incluso en una proporción, valores como 0.5 o 1.0 pueden ser válidos. La detección estadística orienta la revisión; no reemplaza la validación del significado del dato.')

page('Extracción de características de Leads de Piso')
h('Distribución mensual y por asesor')
figure(monthly_fig,'Figura 1. Totales mensuales después de separar cuentas administrativas.',maxheight=2.65)
ptext('Marzo tiene el mayor volumen, con 79 registros (32.38% del total). Abril tiene el menor, con 35 (14.34%). Entre marzo y abril se observa una reducción de 44 registros, equivalente al 55.70%; el archivo no permite establecer si se debe a actividad comercial o diferencias en captura.')
top=leads.sort_values('Total',ascending=False).head(5)
table(['Asesor','Total de leads','Participación'],[(r.Asesor,int(r.Total),f'{r.Total/244*100:.2f}%') for r in top.itertuples()],[4.15,1.2,1.55])
ptext('Alan y Pedro suman 95 registros, el 38.93% del total. Quince asesores tienen al menos un registro. La media es 6.97 por asesor en el periodo y la mediana es cero, lo que refleja la concentración y la presencia de numerosos ceros.')
h('Agrupación mediante Sturges')
ptext('Para los 35 totales por asesor, k = 1 + 3.32 log10(35) = 6.1263, redondeado a seis clases. El rango es 51 y la amplitud final es 8.5. Se usan intervalos cerrados a la derecha, incluyendo cero en la primera clase.')
bins=np.linspace(0,51,7)
counts=pd.cut(leads.Total,bins=bins,include_lowest=True).value_counts(sort=False)
table(['[0, 8.5]','(8.5, 17]','(17, 25.5]','(25.5, 34]','(34, 42.5]','(42.5, 51]'],
      [[int(v) for v in counts]])
ptext('La fila numérica muestra la frecuencia de asesores. El paréntesis excluye el límite inferior y el corchete incluye el límite superior. La primera clase contiene los 20 asesores con cero registros.')

page('Extracción de características de BCS')
h('Cumplimiento de metas por modelo')
code('''bcs["Brecha"] = bcs["Ventas_Reales"] - bcs["Objetivo"]
bcs["Cumplimiento"] = bcs["Ventas_Reales"] / bcs["Objetivo"] * 100
bcs["Participacion"] = (
    bcs["Ventas_Reales"] / bcs["Ventas_Reales"].sum() * 100
)''')
table(['Modelo','Objetivo','Ventas','Brecha','Cumplimiento'],[(r.Modelo,int(r.Objetivo),int(r.Ventas_Reales),f'{int(r.Brecha):+d}',f'{r.Cumplimiento:.2f}%') for r in bcs.itertuples()],[1.4,1.1,1.0,1.0,2.4])
figure(bcs_fig,'Figura 2. Comparación entre las metas por modelo y sus ventas reales.',maxheight=2.7)
ptext('AION, EMKOO y GS4 superan sus metas; EMZOOM, GN8 y GS8 quedan por debajo. EMKOO presenta la mayor brecha positiva, con siete ventas sobre el objetivo, y EMZOOM la mayor brecha negativa, con ocho ventas por debajo.')
ptext('EMZOOM y EMKOO registran 14 ventas cada uno. Juntos representan 28 de 41 ventas, equivalentes al 68.29%. Sin embargo, su cumplimiento difiere: 63.64% y 200%, respectivamente, debido a que las metas asignadas son distintas.')
ptext('Un porcentaje alto con una meta pequeña no implica necesariamente el mayor volumen. AION alcanza 250% con cinco ventas, mientras que EMZOOM registra 14 ventas y no alcanza su meta. Por eso reviso simultáneamente unidades, brecha y cumplimiento.')

page('Regresión lineal simple de Leads de Piso')
h('Relación entre mes y registros de leads')
ptext('Sumo los cuatro meses entre los 35 asesores y codifico enero = 1, febrero = 2, marzo = 3 y abril = 4. El modelo utiliza cuatro observaciones mensuales. La variable independiente es Mes y la dependiente es Leads; no se modelan ventas porque esa variable no forma parte de esta base.')
code('''mensual = pd.DataFrame({
    "Mes": [1, 2, 3, 4],
    "Leads": leads[meses].sum().to_numpy()
})
modelo_leads = LinearRegression().fit(mensual[["Mes"]], mensual["Leads"])
mensual["Prediccion"] = modelo_leads.predict(mensual[["Mes"]])
mensual["Residuo"] = mensual["Leads"] - mensual["Prediccion"]
r2 = modelo_leads.score(mensual[["Mes"]], mensual["Leads"])
r = mensual["Mes"].corr(mensual["Leads"])''')
ptext('Leads estimados = 83.0000 − 8.8000 × Mes')
figure(reg_leads,'Figura 3. Ajuste descriptivo de enero a abril de 2025.',maxheight=2.5)
table(['Mes','Leads','Predicción','Residuo'],[(m,int(y),f'{p:.1f}',f'{e:+.1f}') for m,y,p,e in zip(['Enero','Febrero','Marzo','Abril'],totals,lm['pred'],lm['resid'])],[1.9,1.3,1.9,1.8])
ptext(f'R² = {lm["r2"]:.4f} y r = {lm["r"]:.4f}. La recta explica el 35.07% de la variación de estos cuatro totales. Su pendiente es negativa, pero marzo aumenta respecto de febrero; no existe una disminución uniforme.')
ptext('La muestra de cuatro meses es insuficiente para una previsión anual sólida. El ajuste se evalúa con los mismos datos empleados para estimarlo y no demuestra que el paso del tiempo cause la reducción de registros.')

page('Regresión lineal simple de BCS')
h('Relación entre objetivo y ventas reales')
ptext('Defino Objetivo como variable independiente y Ventas_Reales como variable dependiente. Cada observación es uno de los seis modelos de vehículo. Este ajuste reproduce la relación trabajada en la actividad 5.2, sin reutilizar variables del modelo de leads.')
code('''X_bcs = bcs[["Objetivo"]]
y_bcs = bcs["Ventas_Reales"]
modelo_bcs = LinearRegression().fit(X_bcs, y_bcs)
bcs["Prediccion"] = modelo_bcs.predict(X_bcs)
bcs["Residuo"] = bcs["Ventas_Reales"] - bcs["Prediccion"]
r2_bcs = modelo_bcs.score(X_bcs, y_bcs)
r_bcs = bcs["Objetivo"].corr(bcs["Ventas_Reales"])''')
ptext('Ventas reales estimadas = 3.0546 + 0.515284 × Objetivo')
figure(reg_bcs,'Figura 4. Regresión entre metas y ventas reales de los seis modelos.',maxheight=2.65)
table(['Modelo','Ventas reales','Predicción','Residuo'],[(m,int(y),f'{p:.2f}',f'{e:+.2f}') for m,y,p,e in zip(bcs.Modelo,bcs.Ventas_Reales,bm['pred'],bm['resid'])],[1.7,1.5,1.8,1.9])
ptext(f'R² = {bm["r2"]:.4f} y r = {bm["r"]:.4f}. El modelo explica el 47.46% de la variación observada entre los seis modelos. La relación es positiva, aunque EMKOO vende bastante más de lo que indica la recta y GS8 menos.')
ptext('La pendiente describe una asociación de aproximadamente 0.515 ventas por unidad adicional de objetivo. No significa que elevar una meta provoque ventas: los objetivos pueden haberse fijado según expectativas previas de demanda. El modelo es descriptivo y no tiene validación fuera de muestra.')

page('Análisis de correlación')
h('Correlaciones entre variables originales')
ptext('Calculo Pearson directamente, conservando su signo. Excluyo Total, Predicción, Brecha y Cumplimiento del mapa de variables originales para evitar interpretar relaciones que provienen de sumas o fórmulas.')
code('''corr_leads = leads[meses].corr()
corr_bcs = bcs[["Objetivo", "Ventas_Reales"]].corr()
sns.heatmap(corr_leads, annot=True, vmin=-1, vmax=1,
            cmap="coolwarm", fmt=".2f")
plt.show()
sns.heatmap(corr_bcs, annot=True, vmin=-1, vmax=1,
            cmap="coolwarm", fmt=".2f")
plt.show()''')
figure(corr_fig,'Figura 5. Correlaciones de los conteos por asesor y de los objetivos y ventas por modelo.',maxheight=2.85)
table(['Relación','Pearson r','Observaciones'],[
('Enero y febrero de leads','0.6543','35 asesores'),('Marzo y abril de leads','0.8793','35 asesores'),
('Mes y total mensual de leads','−0.5922','4 meses'),('Objetivo y ventas reales BCS','0.6889','6 modelos')],[3.5,1.2,2.2])
ptext('La correlación entre marzo y abril indica que los asesores con más registros en marzo tienden a registrar más en abril. No contradice la caída del total de abril: una relación entre asesores puede ser positiva aunque el volumen mensual disminuya.')
ptext('Los numerosos asesores con cero en ambos meses influyen en estas relaciones. La matriz mensual por asesor usa 35 filas, mientras que la regresión temporal usa cuatro totales. Son unidades de análisis diferentes y sus coeficientes no deben compararse como si representaran lo mismo.')

page('Hallazgos clave del análisis')
h('Leads de Piso 2025')
ptext('1. La base registra 244 leads entre enero y abril después de separar las cuentas administrativas. Marzo concentra 79 registros y abril 35. Esta variación requiere revisar la cobertura de captura antes de atribuirla a cambios en la demanda.')
ptext('2. Alan Gonzales y Pedro Gonzales reúnen 95 registros, el 38.93% del total. Esta concentración describe el volumen registrado; no permite concluir quién convierte más leads en ventas o atiende con mayor calidad.')
ptext('3. Veinte asesores tienen cero registros en todo el periodo. Conviene distinguir a quienes estaban activos de quienes todavía no ingresaban, ya no trabajaban en la agencia o no registraron su actividad.')
ptext('4. Los cuatro totales más altos por asesor se detectan como atípicos mediante IQR y se conservan. Eliminarlos distorsionaría el volumen registrado sin una razón verificable de error.')
h('BCS')
ptext('5. Los seis modelos suman 41 ventas reales. EMZOOM y EMKOO aportan 14 cada uno y concentran 68.29% de las ventas de esta sección.')
ptext('6. EMKOO supera su meta en siete unidades y EMZOOM queda ocho por debajo. AION tiene el mayor cumplimiento porcentual, con 250%, pero su volumen es de cinco ventas. Las decisiones deben considerar ambos indicadores.')
ptext('7. La suma de objetivos por modelo es 44, frente al objetivo general de 40 en el tablero. El cumplimiento de la sección cambia de 93.18% a 102.50% según el denominador; se necesita una definición única del indicador.')
h('Regresiones')
ptext('8. La regresión de leads sobre Mes obtiene R² = 0.3507 y la de ventas de BCS sobre Objetivo obtiene R² = 0.4746. Ambas describen sus muestras, pero cuatro meses y seis modelos ofrecen evidencia limitada para predecir resultados futuros.')
ptext('9. Las bases cubren unidades y referencias temporales diferentes. No permiten calcular una tasa común de conversión de leads de piso a ventas de BCS ni demostrar que los asesores con más leads expliquen las ventas de esos modelos.')

page('Propuestas y conclusión')
h('Propuestas para el socio formador')
ptext('1. Estandarizar los registros de asesores. Asignar un identificador único y utilizar un catálogo de nombres para evitar duplicados por espacios o variantes. Separar las cuentas administrativas sin perder sus movimientos: deben conservarse en una tabla de control.')
ptext('2. Distinguir los ceros de los datos faltantes. Registrar si el asesor estuvo activo en cada mes y si el periodo tiene captura completa. Un cero debe representar una definición acordada, no una sustitución automática de cualquier celda vacía.')
ptext('3. Revisar el descenso de abril. Comparar los registros con el calendario de operación, las asignaciones de leads y los reportes de captura. Solo después de validar esa información conviene ajustar metas o redistribuir carga de trabajo.')
ptext('4. Conciliar los objetivos de BCS. Definir si la meta oficial es el total general de 40 o la suma por modelos de 44, registrar quién autoriza los cambios y mantener una fecha de corte visible. El tablero debe utilizar el mismo denominador al calcular cumplimiento.')
ptext('5. Analizar brechas por modelo. Revisar las siete ventas adicionales de EMKOO y las ocho faltantes de EMZOOM junto con disponibilidad, promociones y demanda. El resultado puede orientar preguntas comerciales, pero no justifica por sí solo modificar inventario.')
ptext('6. Ampliar y vincular información. Reunir más meses con el mismo criterio de captura y registrar un identificador de lead, asesor, modelo de interés y venta. Esto permitiría estudiar conversión con periodos comparables y evaluar modelos con datos posteriores a los utilizados para entrenarlos.')
h('Conclusión')
ptext('El análisis de Leads de Piso 2025 muestra concentración de registros entre pocos asesores y una reducción importante en abril. BCS muestra diferencias entre las metas y los resultados por modelo, además de una discrepancia entre el objetivo general y el desglosado. La prioridad es mejorar la consistencia de captura y las definiciones de los indicadores antes de convertir las relaciones estadísticas en decisiones comerciales.')
ptext('La extracción, validación de nulos, revisión de atípicos, creación de características y regresión lineal permiten describir ambas bases de forma ordenada. Los resultados se mantienen separados para respetar sus periodos y unidades de observación.')

page('Fuentes y código de apoyo')
h('Bases principales')
ptext('Leads de Piso 2025. Archivo leads_piso_2025_sin_nulos.csv, conservado en Actividad 2.2. Columnas utilizadas: Asesor, ene, feb, mar y abr. Los cálculos de esta evidencia parten de ese archivo ya procesado.')
ptext('BCS. Archivo Analisis GAC Full7agosto 2026( BCS).csv, conservado en Actividad 5.2. Columnas seleccionadas por posición: 1, 3 y 4; filas correspondientes a los seis modelos de vehículo. El notebook Tarea 5.2.ipynb respalda la selección y regresión de BCS.')
h('Ejemplos complementarios')
ptext('Citas Digitales. Tarea_valores_nulos.ipynb, actividad 2.1, respalda el ejemplo de sustitución de nulos. tarea_VA.ipynb, actividad 3.1, respalda el cálculo de IQR y la sustitución por mediana. Estos ejemplos no aportan observaciones a las regresiones principales.')
h('Frecuencias y categorías de leads')
code('''frecuencia = leads[["Asesor", "Total"]].sort_values(
    "Total", ascending=False)
frecuencia["Porcentaje"] = (
    frecuencia["Total"] / frecuencia["Total"].sum() * 100)

k = round(1 + 3.32 * np.log10(len(leads)))
limites = np.linspace(leads["Total"].min(), leads["Total"].max(), k + 1)
leads["Categoria_Total"] = pd.cut(
    leads["Total"], bins=limites, include_lowest=True)
tabla_clases = leads["Categoria_Total"].value_counts(sort=False)''')
h('Exportación de resultados')
code('''leads.to_csv("Leads_Piso_2025_Evidencia.csv", index=False)
bcs.to_csv("BCS_Evidencia.csv", index=False)
mensual.to_csv("Regresion_Leads_Piso_2025.csv", index=False)''')
ptext('Los fragmentos de las secciones principales utilizan pandas, NumPy, matplotlib, seaborn y scikit-learn. Deben ejecutarse con las bases en el directorio de trabajo. El ejemplo de Citas Digitales es independiente y requiere cargar su base antes de usar la variable citas.')
ptext('Las gráficas y estadísticas principales se calcularon a partir de los archivos indicados. Las predicciones se evalúan dentro de la muestra y no constituyen una prueba de desempeño con datos nuevos.')

# Validaciones de los números publicados y estructura del documento.
assert iqr(leads.Total)[4]==4
assert np.allclose(iqr(bcs.Objetivo)[:4],[2.25,8.5,-7.125,17.875])
assert iqr(bcs.Ventas_Reales)[4]==0
assert len(counts)==6 and counts.sum()==35
assert len(doc.inline_shapes)==5
target=ROOT/'Evidencia_Leads_Piso_2025_y_BCS.docx'
doc.save(target)
print(json.dumps(stats,ensure_ascii=False,indent=2))
print(str(target))
