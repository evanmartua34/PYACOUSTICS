import os
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import math 
import scipy
import scipy.signal
import statistics
import time
class struct_1():
	def _init_(self):
		struct_1.a= None
		
class struct_2():
	def _init_(self):
		struct_2.a = None
		
class opt_class(): 
	def _init_(self, fs, verbose):
		self.fs = None
		self.verbose = None

class lmopt():
	def _init_(self,a):
		self.a = None


def smooth(x,window_len=11,window='hanning'):
	import numpy
	"""smooth the data using a window with requested size.
    
	This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
	x= np.array(x)
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
		
	if (x.size < window_len):
		raise ValueError("Input vector needs to be bigger than window size.")

	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


	s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=numpy.ones(window_len,'d')
	else:
		w=eval('numpy.'+window+'(window_len)')

	y=numpy.convolve(w/w.sum(),s,mode='valid')
	return y


def smooth_demo():
	from numpy import *
	from pylab import *
	t=linspace(-4,4,100)
	x=sin(t)
	xn=x+randn(len(t))*0.1
	y=smooth(x)

	ws=31

	subplot(211)
	plot(ones(ws))

	windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

	hold(True)
	for w in windows[1:]:
		eval('plot('+w+'(ws) )')

	axis([0,30,0,1.1])

	legend(windows)
	title("The smoothing windows")
	subplot(212)
	plot(x)
	plot(xn)
	for w in windows:
		plot(smooth(xn,10,w))
	l=['original signal', 'signal with noise']
	l.extend(windows)

	legend(l)
	title("Smoothing a noisy signal")
	show()
		

def detect_syllable_nuclei(path_to_files, output_path):
	files=[]
	for filecek in os.listdir(path_to_files):
		if filecek.endswith(".wav"):
			files.append(filecek)
			
	for i in range(0,len(files)):
		name , tossExt = os.path.splitext(files[i])
		tossPath = path_to_files
		y,fs = sf.read(os.path.join(path_to_files,files[i]))
		#opt = opt_class()
		opt = {}
		opt['fs']=fs
		opt['verbose']=0
		sn = fu_sylncl(y,opt)
		sn = np.array(sn)
		sn.astype(float)
		sn = sn / fs
		sn.tolist()
		output_fn = output_path +'/'+name + '.txt'
		fd = open(output_fn,"w+")
		for i in range(0,len(sn)):
			print(sn[i])
			tulis = float(sn[i])
			fd.write("\n %f" %tulis)
		fd.close()

def fu_sylncl(s,opt):
	#if nargin ==1; opt=struct; end	
	opt = fu_optstruct_init(opt , ['do', 'nouse_int', 'do_nouse', 'errtype'] , ['apply', [], 2, 'f'])
	ofld=['do', 'bf', 'f_thresh', 'length', 'rlength', 'md', 'e_min', 'fs', 'verbose', 'pau', 'unv']
	
	opt['nouse_int'] = ' '
	
	if (opt['do']=='apply'):
		struct_1 = {}
		struct_2 = {}
 		odef=['apply', [212.5509, 3967.1], 1.0681, 0.0776, 0.1491, 0.1, 0.1571, 16000, 0, struct_1, struct_2]
		
		#odef={'apply' [212.5509 3967.1] 1.0681 0.0776 0.1491 0.1 0.1571 16000 0 }
		opt=fu_optstruct_init(opt,ofld,odef)
		opt['pau'] = fu_optstruct_init(opt['pau'], ['fs', 'ret'], [opt['fs'], 'smpl'])
		opt['unv'] = fu_optstruct_init(opt['unv'], ['sts'], [1])
		sn=fu_sylncl_sub(s,opt)
		#if (nargout>1)
		#sb=fu_sylbnd(s,sn,opt)
	
	else:
		##s_glob=s
		##opt_glob=opt
		#o_opt=optimset(@fminsearch);
		#o_opt=optimset('LargeScale','on');
		w0=[2.3, 2.9, 1.06, 0.08, 0.14, 0.16]
		#[w fval ef o]=fminsearch(@fu_sylncl_err,w0,o_opt);
		opt=fu_optstruct_init(opt,ofld,['apply', [w(1)*100, w(2)*1000], w(3), w(4), w(5), w(6), opt['fs'], 1])
		sn=fu_sylncl_sub(s,opt)
		
	if opt['verbose']==1 :
		t=[]
		for x in range (1,len(s)+1):
			t.append(x)
		plt.plot(t,s)
		#for i in sn; 
			#plot([i i],[-1 1],'-r'); end
		# if nargout>1
		#satu = np.arange(-1, 1)
		#i2=[]
		#for i in sb:
		#	i2.append(i)
		#	i3=np.array(i2)
		#	plt.plot(i3,satu,'-g')

	if (opt['do'] == 'train'):
		opt['do']='apply'
		opt['error']=fval
		sn = opt
		sn['opt'] = opt
		#save('sn_opt','sn_opt');
	return sn #,sb

	
def fu_optstruct_init(opt, optfields, optdefaults):
	for n in range(0,len(optfields)):
		if not(optfields[n] in opt):
			if (not(type(optdefaults[n]) == int or type(optdefaults[n]) == float)  ) and (optdefaults[n] == 'oblig'):
				raise ValueError('opt field "%s" has to be defined by the user!',optfields[n])
			opt[optfields[n]] = optdefaults[n]
	return opt
	
def fu_sylncl_sub(s,opt):
	rws = math.floor(opt['rlength'] * opt['fs'])
	ls = len(s)
	ml = math.floor(opt['length'] * opt['fs'])
	md = math.floor(opt['md'] * opt['fs'])
	sts = max(1, math.floor(0.03 * opt['fs']))
	stsh = math.floor(sts/2)
	
	t_nou_init = np.array([])
	t_nou_pau =np.array([])
	voi = np.array([])
	t_nou = np.array([[]])

	if 'nouse_int'in opt:
		t_nou_init = np.array([[opt['nouse_int']]])
	if opt['do_nouse']>0:
		if (opt['do_nouse'] < 3):
			t_nou_pau = np.array(fu_pause_detector(s,opt['pau']))
			
		if (opt['do_nouse'] == 1 or opt['do_nouse']==3):
			voi,zrr = fu_voicing(s, opt['fs'], opt['unv'])
			
	for i in range(0,t_nou_init.shape[0]):
		#t_nou = [t_nou, t_nou_init_init[i,1:2]]  #line 40 
		#t_nou.append(t_nou_init[i,0:1])
		
		if t_nou_init.shape[1]>1:
			t_nou = np.column_stack((t_nou, t_nou_init[i,0:2]))
	
	for i in range(0,t_nou_pau.shape[0]):
		#t_nou = [t_nou, t_nou_pau[i,1:2]]
		#t_nou.append(t_nou_pau[i,0:1]) #cek lagi
		
		t_pau_tambah = np.arange(t_nou_pau[i,0],t_nou_pau[i,1]+1)
		t_pau_tambah.tolist()
		t_pau_tambah = np.array([t_pau_tambah])
		t_nou = np.column_stack((t_nou, t_pau_tambah))
	tambah = np.where(voi==0)
	tambahbaru = tambah[0]
	tambahtrans = tambahbaru.reshape(1,len(tambahbaru))
	#tambah  = np.transpose(tambah)
	np.column_stack((t_nou,tambah))
	
	t_nou = np.unique(np.transpose(np.array(t_nou)))
	if len(opt['bf']) == 1 :
		ft='low'
	else :
		ft='band'
	
	ord=5
	s=fu_filter(s,ft,opt['bf'],opt['fs'],ord)
	e_y=[]
	jarak= np.arange(0,ls,sts )
	for i in jarak :
		yi= np.arange(i,min(ls,i+ml-1))
		yi=[int(k) for k in yi]
		y = [s[k] for k in yi]
		e_y.append(fu_rmse(y))
	e_min = opt['e_min'] * max(e_y)
	mey = max(e_y)
	
	t=[]
	
	all_i = []
	all_e = []
	all_r = []
	i2=[]
	e_y2 = []
	e_rw2 = []
	for i in jarak :
		yi = fu_i_window(i,ml,ls)
		yi = [int(k) for k in yi]
		y = [s[k] for k in yi]
		e_y  = fu_rmse(y)
		rwi = fu_i_window(i,rws,ls)
		rwi = [int(n) for n in rwi]
		rw = [s[k] for k in rwi]
		e_rw = fu_rmse(rw)
		i2.append(i)
		e_y2.append(e_y)
		e_rw2.append(e_rw)
	
	all_i = np.array(i2)
	all_e = np.array(e_y2)
	all_r = np.array(e_rw2)
	
	#lmopt =struct 
	lmopt = {}
	lmopt['peakmpd'] = math.floor(opt['fs'] * opt['md']/ sts)
	pks, idx = fu_locmax(all_e, lmopt)
	t = []
	
	
	for i in idx:
		if ((all_e[i-1] >= ((all_r[i-1]) * (opt['f_thresh']))) and (all_e[i-1] > e_min)):
			cek = np.where(t_nou == all_i[i-1])
			if len(cek[0]) == 0 :
				t.append(all_i[i-1])
	t=np.array(t)
	
	print('cek T')
	print('====================')
	print(t)
	print('====================')
	return t 
	
	
def fu_filter(s, t, gf, fs, o):
	if (type(gf) == list) or (type(fs) == list):
		gf = np.array(gf)
		fs = np.array(fs)
		fn = gf / (fs/2)
		cekfil= np.ones(fn.shape)
	
		if (sum(np.greater_equal(fn, cekfil))):
			sflt = s
	#elif (fn>=1):
		#sflt = s
		else:
		#if nargin < 5; o=5; end
			#if (t == 'band'):
			#	b, a = scipy.signal.butter(o,fn)
			#else:
			b, a = scipy.signal.butter(o, fn ,t)
			sflt = np.array(scipy.signal.filtfilt(b,a,s))
			if len(np.isnan(sflt)) > 0:
				print('filtering not possible, returning original signal')
				sflt = s 
	else :
		fn = gf / (fs/2)
		if (fn>=1):
			sflt = s
		else:
			#if nargin < 5; o=5; end
			#if (t == 'band'):
				#b, a = scipy.signal.butter(o,fn)
			#else:
			b, a = scipy.signal.butter(o, fn ,t)
			sflt = np.array(scipy.signal.filtfilt(b,a,s))
			if len(np.isnan(sflt)) > 0:
				print('filtering not possible, returning original signal')
				sflt = s	
	return sflt
	
def fu_sylbnd(s, sn ,opt):
	ml = math.floor(opt['length'] * opt['fs'])
	sts = max(1, math.floor(0.03 *opt['fs']))
	sb = []
	for i in range (0,len(sn)-1):
		on = sn[i]
		off = sn[i+1]
		sw = s[on:off]
		ls = len(sw)
		all_i = np.array([])
		all_e = np.array([])
		j2=[]
		e_y2=[]
		jarak= np.arange(1,len(sw)+sts ,sts)
		for j in jarak :
			yi = fu_i_window(j,ml,ls)
			y = sw[yi]
			e_y = fu_rmse(y)
			e_y2.append(e_y)
			j2.append(j)
			
		e_y2  = np.array(e_y2)
		all_e = np.concatenate((all_e,e_y2))
		j2    = np.array(j2)
		all_i = np.concatenate((all_i,j2))
		ymin  = min(all_e)
		ymini = np.where(all_e = ymin)
		sb.append(on+all_i[ymini(1)])
	sb = np.array(sb);
		
	return sb
	
def fu_locmax(y,opt):
	#if nargin<2; opt=struct; end
	struct_3 = {}
	struct_4 = {}
	print('cek size')
	print(y.shape)
	opt = fu_optstruct_init(opt,['smooth', 'peak'], [struct_3, struct_4])
	opt['smooth'] = fu_optstruct_init(opt['smooth'], ['win', 'mtd', 'order'], [float(1), 'None', float(1)])
	opt['peak']=fu_optstruct_init(opt['peak'],['mph', 'th', 'mpd'],[float('-inf'), float(0), float(1)]);
	opt['peak']['mpd'] = min(opt['peak']['mpd'] , float(len(y)-1))
	idx = scipy.signal.find_peaks(fu_smooth(y,opt['smooth']),distance = opt['peak']['mpd'], height = opt['peak']['mph'], threshold = opt['peak']['th'])
	ys = fu_smooth(y,opt['smooth'])
	idx = idx[0]
	
	print('cek locmax')
	print('------------------')
	print(ys.shape)
	print(idx)
	print('------------------')

	idx = [int(n) for n in idx]
	pks = [ys[k-1] for k in idx]
	
	if len(pks)==0 :
		idx = scipy.signal.find_peaks(y)
		pks = [y[k] for k in idx]
	y2= y 
	y2.tolist()
	y2 = np.array([y2])
	if ((y2.shape)[1]==1):
		pks = fu_r2c(pks)
		idx = fu_r2c(idx)
	return pks, idx
	
def fu_rmse(*args):
	if len(args) < 2:
		x=np.array(args)
		x.astype(float)
		e= math.sqrt(np.sum(x**2)/len(x[0]));
	else:
		x=np.array(args[0])
		y=np.array(args[1])
		x.astype(float)
		y.astype(float)
		e = math.sqrt(np.sum((x-y)**2)/len(x));

	return e

def fu_i_window(i,wl,l):
	hwl = math.floor(wl/2)
	a = max(i-hwl,1)
	b = min(i+hwl,l)
	wi = np.arange(a,b+1)
	d= wl - len(wi)
	if (d>0):
		if wi[0]>1:
			o = max(wi[0]-d, 1)
			wi = np.arange(o, wi[-1])
			d = wl - len(wi)
		if d>0:
			if wi[-1]<l:
				o = min(wi[-1]+d , l)
				wi = np.arange(wi[0],o)
	wi = wi-1 
	return wi
		
def fu_smooth(y,opt):
	#if nargin<1; opt=struct; end
	#if opt['mtd'] == 'None':
	opt = fu_optstruct_init(opt,['mtd', 'wl', 'order'],['mova', 5, 3])
	#mova
	#else :
	#	opt = fu_optstruct_init(opt,['mtd', 'wl', 'order'],['mova', 5, 3])
	if (opt['mtd'] == 'None'):
		ys = y
	elif (not(opt.mtd == 'sgolay')):
		ys = smooth(y,opt.wl, opt.mtd)
	else:
		#ys = smooth(y, opt.wl, opt.mtd , opt.order)	
		ys = smooth(y)
	return ys
	
def fu_r2c(v):
	tb=0
	if (v.shape[0]==1):
		v= v.transpose
		tb=1
	
	#if nargout==2
		t=tb
	return v,t
		
def fu_voicing(y,sr,opt):
	opt = fu_optstruct_init(opt,['wl', 'th', 'sts', 'zr_th', 'do', 'min_nf', 'ret'],[0.03, 0.002, 0.01, 2000, 'apply', 3, 'w']);
	opt[sr] = sr
	
	if (opt[do] == 'apply' ):
		voi , zr = fu_voicing_sub(y,opt)
	#else :
		#%o_opt=optimset(@fminunc);
		#o_opt=optimset(@fminsearch);
		#o_opt=optimset('LargeScale','on');
		#w0=[0.004 1000];
		#%[w fval ef o]=fminunc(opt.errfun,w0,o_opt);
		#[w fval ef o]=fminsearch(opt.errfun,w0,o_opt);
		#opt.th=w(1);
		#opt.zr_th=w(2);
		#[voiv zr] = fu_voicing_sub(y,opt);
		#% error
		#voiv=fu_trim_vec(voiv,opt.ref,0);
		#e = pdist([voiv;opt.ref],'hamming');
		#voi=opt;
		#voi.err=e;
	return voi,zrr

def fu_voicing_sub(y,opt):
	zr = fu_zero_crossing_rate(y, opt[sr], opt)
	voi = np.zeros(len(zr))
	voi[np.where((zr<opt[zr_th]) and (zr>0))] = 1
	if (opt[min_nf] > 1):
		voi = fu_smooth_binvec(voi, opt[min_nf])
	
	#if nargout==2; zrr=zr; end
	return voi, zrr
	
def fu_smooth_binvec(v, b, l):
	r = abs(b-1)
	b = np.array(v)
	i = np.where(v==b)
	#if length(i)==0; return; end
	di = []
	di.append(1)
	di.append(diff(i))
	seq_i = []
	for j in range(1,len(di)+1):
		if (di(j) >1):
			if len(seq_i) < l: 
				v[seq_i] == r
			seq_i = [] 
		seq_i.append(i[j])
		
	if len(seq_i) < l: 
		v[seq_i] = r
	return v

def fu_pause_detector(s,opt):
	ofld=['f_thresh', 'length', 'rlength', 'fs', 'ret']
	odef=[0.0767, 0.1524, 5, 16000, 's']
	opt=fu_optstruct_init(opt,ofld,odef)
	#print('shape S')
	#print('--------------')
	#s.tolist()
	#s = np.array([s])
	#print(s.shape)
	#print('---------------')
	s = s- np.mean(s)
	s = fu_filter(s,'low',10000, opt['fs'],5)
	rws = math.floor(opt['rlength'] * opt['fs'])
	ls = len(s)

	ml = math.floor(opt['length'] * opt['fs'])
	e_glob = fu_rmse(s)
	
	
	t_glob = opt['f_thresh'] * e_glob
	sts = max(1, math.floor(0.05 * opt['fs']))
	
	stsh = math.floor(sts/2)
	t = np.array([])
	
	j = 1
	jarak = np.arange(0,ls,sts)
	len(jarak)
	for i in jarak :
		i = i+1
		yi = np.arange(i,(min(ls,i+ml - 1))+1)
		yi = [int(n) for n in yi]
		y = [s[k-1] for k in yi]
		
		e_y = fu_rmse(y)
		
		rwi = fu_i_window(min(i+stsh,ls),rws,ls)
		rwi = [int(n) for n in rwi]
		rw = [s[k] for k in rwi]
		
		#rw = s[fu_i_window(min(i+stsh,ls),rws,ls)]
		e_rw = fu_rmse(rw)
		if e_y <= e_rw * opt['f_thresh']:
			
			if(t.shape[0] ==j):
				
				if yi[0] < t[j-1][1]:
					#t[j,1] =  yi(-1)
					#np.row_stack((t[j-1][1],yi(-1))
					t[j-1][1] = yi[-1]
				else:
					j = j+1
					tambahan_baris= np.column_stack((yi[0] , yi[-1]))
					#t[j-1,0:2] = np.column_stack((yi[0] , yi[-1]))
					t = np.row_stack((t,tambahan_baris))
			else:
				
				ytambah = [yi[0]]
				ytambah.append(yi[-1]) 
				ytambah = np.array([ytambah])
				
				
				#t[j,:] = ytambah
				if (t.shape[0] ==0):
					t = ytambah
					print('nilai t')
					print(t)
				else:
					if(t.shape[1] == 0):
						t = ytambah
					np.column_stack((t,ytambah))
				
				#t[j,:] = [yi[1], yi[-1]]
	
	if (opt['ret'] == 's'):
		t=t/opt['fs']
	return t