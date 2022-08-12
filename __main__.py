import random
import re
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import jax
import jax.numpy as jnp
import threading
import configparser
from os import path, getcwd, mkdir
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from dalle_mini import DalleBartProcessor
from flax.training.common_utils import shard_prng_key
from functools import partial
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageTk
from datetime import datetime

gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0
DALLE_MODEL = path.join(getcwd(), 'dalle-mini')
DALLE_COMMIT_ID = None
VQGAN_REPO = path.join(getcwd(), 'vqgan_imagenet_f16_16384')
VQGAN_COMMIT_ID = None
firstrun = True
buttons = []
borders = []
imgtk = []
images2 = []
filepath = None
collage = None
prompts = []
current = 0
cancel = False

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
	return model.generate(**tokenized_prompt,prng_key=key,params=params,top_k=top_k,top_p=top_p,temperature=temperature,condition_scale=condition_scale)

@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
	return vqgan.decode_code(indices, params=params)

def main():
	global firstrun
	if(firstrun):
		firstrun = False
		global model, params, vqgan, vqgan_params, processor
		model, params = DalleBart.from_pretrained(DALLE_MODEL, dtype=jnp.float16, _do_init=False)
		vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, _do_init=False)
		params = replicate(params)
		vqgan_params = replicate(vqgan_params)
		processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

def main2(launch):
	if(launch):
		main()

class StationeryFunctions:
	def __init__(self, text):
		self.text = text
		self.create_binding_keys()
		self.binding_functions_config()
		self.join_function_with_main_stream()

	def join_function_with_main_stream(self):
		self.text.storeobj['Copy']   =  self.copy
		self.text.storeobj['Cut']    =  self.cut
		self.text.storeobj['Paste']  =  self.paste
		self.text.storeobj['Undo']   =  self.undo
		self.text.storeobj['Redo']   =  self.redo 
		self.text.storeobj['SelectAll']=self.select_all
		self.text.storeobj['DeselectAll']=self.deselect_all
		return

	def binding_functions_config(self):
		self.text.tag_configure("sel", background="skyblue")
		self.text.configure(undo=True,autoseparators=True, maxundo=-1)
		return

	def copy(self, event=None):
		self.text.event_generate("<<Copy>>")
		return

	def paste(self, event=None):
		self.text.event_generate("<<Paste>>")
		return

	def cut(self, event=None):
		self.text.event_generate("<<Cut>>")
		return

	def undo(self, event=None):
		self.text.event_generate("<<Undo>>")
		return

	def redo(self, event=None):
		self.text.event_generate("<<Redo>>")
		return

	def create_binding_keys(self):
		for key in ["<Control-a>","<Control-A>"]:
			self.text.master.bind(key, self.select_all)
		for key in ["<Button-1>","<Return>"]:
			self.text.master.bind(key, self.deselect_all)
		return

	def select_all(self, event=None):
		self.text.tag_add("sel",'1.0','end')
		return

	def deselect_all(self, event=None):
		self.text.tag_remove("sel",'1.0','end')
		return

class TtkScale(ttk.Frame):
    def __init__(self, master=None, **kwargs):
        ttk.Frame.__init__(self, master)
        self.columnconfigure(0, weight=1)
        self.showvalue = kwargs.pop('showvalue', True)
        self.tickinterval = kwargs.pop('tickinterval', 0)
        self.digits = kwargs.pop('digits', '0')
        
        if 'command' in kwargs:
            fct = kwargs['command']
            
            def cmd(value):
                fct(value)
                self.display_value(value)
                
            kwargs['command'] = cmd
        else:
            kwargs['command'] = self.display_value
            
        self.scale = ttk.Scale(self, **kwargs)
        
        style = ttk.Style(self)
        style_name = kwargs.get('style', '%s.TScale' % (str(self.scale.cget('orient')).capitalize()))
        self.sliderlength = style.lookup(style_name, 'sliderlength', default=30)
        
        self.extent = kwargs['to'] - kwargs['from_']
        self.start = kwargs['from_']
        if self.showvalue:
            ttk.Label(self, text=' ').grid(row=0)
            self.label = ttk.Label(self, text='0')
            self.label.place(in_=self.scale, bordermode='outside', x=0, y=0, anchor='s')
            self.display_value(self.scale.get())
            
        self.scale.grid(row=1, sticky='ew')
        
        if self.tickinterval:
            ttk.Label(self, text=' ').grid(row=2)
            self.ticks = []
            self.ticklabels = []
            nb_interv = round(self.extent/self.tickinterval)
            formatter = '{:.' + str(self.digits) + 'f}'
            for i in range(nb_interv + 1):
                tick = kwargs['from_'] + i * self.tickinterval
                self.ticks.append(tick)
                self.ticklabels.append(ttk.Label(self, text=formatter.format(tick)))
                self.ticklabels[i].place(in_=self.scale, bordermode='outside', x=0, rely=1, anchor='n')
            self.place_ticks()

        self.scale.bind('<Configure>', self.on_configure)
        
    def convert_to_pixels(self, value):
        return ((value - self.start)/ self.extent) * (self.scale.winfo_width()- self.sliderlength) + self.sliderlength / 2
        
    def display_value(self, value):
        x = self.convert_to_pixels(float(value))
        half_width = self.label.winfo_width() / 2
        if x + half_width > self.scale.winfo_width():
            x = self.scale.winfo_width() - half_width
        elif x - half_width < 0:
            x = half_width
        self.label.place_configure(x=x)
        formatter = '{:.' + str(self.digits) + 'f}'
        self.label.configure(text=formatter.format(float(value)))
    
    def place_ticks(self):
        tick = self.ticks[0]
        label = self.ticklabels[0]
        x = self.convert_to_pixels(tick)
        half_width = label.winfo_width() / 2
        if x - half_width < 0:
            x = half_width
        label.place_configure(x=x)
        for tick, label in zip(self.ticks[1:-1], self.ticklabels[1:-1]):
            x = self.convert_to_pixels(tick)
            label.place_configure(x=x)
        tick = self.ticks[-1]
        label = self.ticklabels[-1]
        x = self.convert_to_pixels(tick)
        half_width = label.winfo_width() / 2
        if x + half_width > self.scale.winfo_width():
            x = self.scale.winfo_width() - half_width
        label.place_configure(x=x)
        
    def on_configure(self, event):
        """Redisplay the ticks and the label so that they adapt to the new size of the scale."""
        self.display_value(self.scale.get())
        self.place_ticks()

    def get(self):
        if self.digits >= 0:
            return round(self.scale.get(), self.digits)
        else:
            return self.scale.get()

    def set(self, val):
        if self.digits >= 0:
            return self.scale.set(round(val, self.digits))
        else:
            return self.scale.set(val)

def try_or(num, default=None, expected_exc=(Exception,)):
	try:
		num = float(num)
		if(num == 0.0 or num == 0):
			return default
		else:
			return num
	except expected_exc:
		return default

def makepath(p):
	filepath = 'out/' + re.sub(r'\W+', '', p[:20])
	if(not path.exists(filepath)):
		mkdir(filepath)
	return filepath

def start(promptsin, colsin, runsin, temperature, gen_top_p, gen_top_k, cond_scale):
	global model, params, vqgan, vqgan_params, processor, DALLE_MODEL, DALLE_COMMIT_ID, VQGAN_REPO, VQGAN_COMMIT_ID, imgtk, filepath, prompts, current, nums, cancel
	main()
	n_predictions = 3*colsin
	nums = np.arange(1, n_predictions+1).reshape(colsin,3)
	current = len(imgtk)-1
	for ep in range(runsin):
		for prompt in promptsin:
			if(cancel):
				if(len(borders) > 0):
					borders[-1].config(highlightbackground="yellow")
				break
			else:
				if(prompt != "" and len(prompt) > 1):
					current += 1
					print('Generating "' + prompt + '"...')
					prompts.append(prompt)
					selectline()
					seed = random.randint(0, 2**32 - 1)
					key = jax.random.PRNGKey(seed)
					tokenized_prompts = processor([prompt])
					tokenized_prompt = replicate(tokenized_prompts)
					imgtk.append([])
					images2.append([])
					key, subkey = jax.random.split(key)
					encoded_images = p_generate(tokenized_prompt,shard_prng_key(subkey),params,gen_top_k,gen_top_p,temperature,cond_scale)
					encoded_images = encoded_images.sequences[..., 1:]
					decoded_images = p_decode(encoded_images, vqgan_params)
					decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
					for i in range(n_predictions):
						if(cancel):
							if(len(borders) > 0):
								borders[-1].config(highlightbackground="yellow")
							break
						else:
							key, subkey = jax.random.split(key)
							encoded_images = p_generate(tokenized_prompt,shard_prng_key(subkey),params,gen_top_k,gen_top_p,temperature,cond_scale)
							encoded_images = encoded_images.sequences[..., 1:]
							decoded_images = p_decode(encoded_images, vqgan_params)
							decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
							for decoded_img in decoded_images:
								img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
								images2[-1].append(img)
								addbutton(img, i)
					print('Generation complete.\n')

def start2(entry1, entry2, entry3):
	global button2, cancel
	check_changed()
	if(button2.cget('text') == "Cancel"):
		cancel = True
		button2.config(text="Brute Force Common Parameters")
		print('Cancelled.\n')
	else:
		cancel = False
		button2.config(text="Cancel")
		start(entry1, entry2, entry3, 0.5, 1.0, None, 10.0)
		start(entry1, entry2, entry3, 1.0, 0.5, None, 10.0)
		start(entry1, entry2, entry3, 0.5, 0.5, None, 10.0)
		start(entry1, entry2, entry3, None, None, None, 10.0)
		#start(entry1, entry2, entry3, 0.1, 1.0, None, 10.0)
		#start(entry1, entry2, entry3, 1.0, 0.1, None, 10.0)
		button2.config(text="Brute Force Common Parameters")
		print('All generations completed.\n')

def start3(entry1, entry2, entry3, entry4, entry5, entry6, entry7):
	global button, cancel
	check_changed()
	if(button.cget('text') == "Cancel"):
		cancel = True
		button.config(text="Start")
		print('Cancelled.\n')
	else:
		cancel = False
		button.config(text="Cancel")
		start(entry1, entry2, entry3, entry4, entry5, entry6, entry7)
		button.config(text="Start")
		print('All generations completed.\n')

def selectline():
	global promptlabel, promptframe, current, entry1
	promptlabel.config(text = "Prompt: " + prompts[current])
	i = 1
	for line in entry1.get('1.0', tk.END).splitlines():
		if(line == prompts[current]):
			entry1.tag_remove("cline", "1.0", "end")
			entry1.tag_add("cline", str(i) + ".0", str(i) + ".0 lineend")
			entry1.tag_configure("cline", background="OliveDrab1", foreground="black")
			break
		i += 1

def pathcheck():
	global current
	if(current < len(prompts)):
		global filepath
		filepath = makepath(prompts[current])

def saveimg(image, collage=False):
	pathcheck()
	global filepath
	if(collage):
		image.save(filepath + '/out.png')
	else:
		datenow = datetime.now().strftime("%d-%m-%Y-%H-%M")
		if(path.exists(filepath + '/' + datenow + '.png')):
			savenum = None
			for i in range(50):
				savenum = str(i+1)
				if(not path.exists(filepath + '/' + datenow + '_' + savenum + '.png')):
					break
			image.save(filepath + '/' + datenow + '_' + savenum + '.png')
		else:
			image.save(filepath + '/' + datenow + '.png')

def addbutton(img, i):
	global buttons, borders, donelist, nums, picsframe, entry2
	buts = 3*int(entry2.get())
	if(len(buttons)-1 >= buts):
		buttons[-buts].destroy()
		del buttons[-buts]
		borders[-buts].destroy()
		del borders[-buts]
	tkimgtk = ImageTk.PhotoImage(img, master=picsframe)
	imgtk[-1].append(tkimgtk)
	#for b in borders:
	#	b.config(highlightbackground = "white")
	if(len(borders) > 0):
		borders[-1].config(highlightbackground = "white")
	borders.append(tk.Frame(picsframe, highlightbackground = "black", highlightthickness = 2, bd=0))
	buttons.append(ttk.Button(borders[-1], image=tkimgtk, command=lambda image=img: saveimg(image)))
	buttons[-1].pack()
	r = 1
	c = 1
	num = 0
	for a in nums:
		for j in a:
			if(num == i):
				borders[-1].grid(row=r, column=c, in_=picsframe)
			c+=1
			num+=1
		c = 1
		r+=1

def move(prev=False):
	global imgtk,current,images2
	dochange = False
	if(prev):
		if(current > 0):
			current -= 1
			dochange = True
	elif(current < len(imgtk)-1):
		current += 1
		dochange = True
	if(dochange):
		selectline()
		i = 0
		for img in images2[current]:
			addbutton(img, i)
			i+=1

def savecollage():
	global entry1, entry2, imgtk, images2
	x = np.arange(0, len(imgtk[current])).reshape(int(entry2.get()), 3)
	Horizontali = []
	for i in range(int(entry2.get())):
		Horizontali.append(np.hstack([ImageOps.expand(images2[current][x[i][0]],border=4,fill='white'),ImageOps.expand(images2[current][x[i][1]],border=4,fill='white'),ImageOps.expand(images2[current][x[i][2]],border=4,fill='white')]))
	Vertical_attachment = np.vstack(Horizontali)
	imgvert = Image.open('template3x' + str(int(entry2.get())) + '.png')
	width, height = imgvert.size
	imgvert.paste(Image.fromarray(Vertical_attachment), (19, 139))
	txtimgbase = Image.new("RGB", (680, 30), (255, 255, 255))
	txtimg = ImageDraw.Draw(txtimgbase)
	myFont = ImageFont.truetype('SourceSansPro-Regular.ttf', 20)
	txtimg.text((0, 0), entry1.get(str(current+1) + ".0", str(current+1) + ".0 lineend"), font=myFont, fill =(55, 65, 81))
	imgvert.paste(txtimgbase, (37, 80))
	saveimg(imgvert, True)

def check_changed():
	global config, checkval, entry1, entry2, entry3, entry4, entry5, entry6, entry7
	config['model'] = {}
	config['model']['preload'] = str(checkval.get())
	config['model']['prompt'] = entry1.get("1.0", "end-1c")
	config['model']['cols'] = str(int(entry2.get()))
	config['model']['iterations'] = entry3.get()
	config['model']['temperature'] = str(entry4.get())
	config['model']['top_p'] = str(entry5.get())
	config['model']['top_k'] = str(entry6.get())
	config['model']['cond_scale'] = str(entry7.get())
	with open('config.ini', 'w') as configfile:
		config.write(configfile)

if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('config.ini')
	pics = tk.Tk()
	pics.title("DALL-E Mini Generator")
	pics.style = ttk.Style(pics)
	pics.style.theme_use('vista')
	pics.style.configure('my.Horizontal.TScale', sliderlength=20)
	picsframe=ttk.Frame(pics)
	window=ttk.Frame(pics)
	textframe=ttk.Frame(pics)
	butframe=ttk.Frame(pics)
	butframe2=ttk.Frame(pics)
	promptframe=ttk.Frame(pics)
	label1 = ttk.Label(textframe, text="Prompts (separate by new line)")
	scroll = ttk.Scrollbar(textframe)
	entry1 = tk.Text(textframe, height=10, width=200, yscrollcommand=scroll.set, undo=True)
	scroll.config(command=entry1.yview)
	entry1.storeobj = {}
	StationeryFunctions(entry1)
	entry1.insert(tk.END, "")
	label2 = ttk.Label(window, text="Cols")
	entry2 = TtkScale(window, style='my.Horizontal.TScale', from_=1, to=5, orient='horizontal', length=300, tickinterval=1, digits=0)
	label3 = ttk.Label(window, text="Iterations:")
	entry3 = ttk.Entry(window, width=5)
	entry3.insert(0, "1")
	label4 = ttk.Label(window, text="Temperature")
	entry4 = TtkScale(window, style='my.Horizontal.TScale', from_=0.0, to=1.0, orient='horizontal', length=300, tickinterval=0.1, digits=2)
	label5 = tk.Label(window, text="Top P")
	entry5 = TtkScale(window, style='my.Horizontal.TScale', from_=0.0, to=1.0, orient='horizontal', length=300, tickinterval=0.1, digits=2)
	label6 = tk.Label(window, text="Top K")
	entry6 = TtkScale(window, style='my.Horizontal.TScale', from_=0, to=100, orient='horizontal', length=300, tickinterval=10, digits=0)
	label7 = tk.Label(window, text="Cond Scale")
	entry7 = TtkScale(window, style='my.Horizontal.TScale', from_=0.0, to=100.0, orient='horizontal', length=300, tickinterval=10, digits=2)
	entry2.set(3)
	entry4.set(0.50)
	entry5.set(0.50)
	entry6.set(0)
	entry7.set(10.0)
	button = ttk.Button(
		window,
		text="Start",
		command=lambda: threading.Thread(target=start3, args=(entry1.get("1.0", "end-1c").split("\n"), int(entry2.get()), int(entry3.get()), try_or(entry4.get()), try_or(entry5.get()), try_or(entry6.get()), try_or(entry7.get()))).start()
	)
	button2 = ttk.Button(
		window,
		text="Brute Force Common Parameters",
		command=lambda: threading.Thread(target=start2, args=(entry1.get("1.0", "end-1c").split("\n"), int(entry2.get()), int(entry3.get()))).start()
	)
	button3 = ttk.Button(
		butframe2,
		text="Save Collage",
		command=savecollage
	)
	button4 = ttk.Button(
		butframe,
		text="<",
		command=lambda: move(True)
	)
	button5 = ttk.Button(
		butframe,
		text=">",
		command=lambda: move()
	)
	promptlabel = tk.Message(promptframe, text="", width=1000)
	checkval = tk.BooleanVar(window)
	check = ttk.Checkbutton(window, text="Preload model on startup?", variable=checkval, command=check_changed)
	if('model' in config):
		checkval.set(config['model'].getboolean('preload', fallback=False))
		entry1.delete(1.0,"end")
		entry1.insert(1.0, config['model'].get('prompt', fallback=''))
		entry2.set(config['model'].getint('cols', fallback=3))
		entry3.delete(0,"end")
		entry3.insert(0, config['model'].get('iterations', fallback=''))
		entry4.set(config['model'].getfloat('temperature', fallback=0.50))
		entry5.set(config['model'].getfloat('top_p', fallback=0.50))
		entry6.set(config['model'].getfloat('top_k', fallback=0))
		entry7.set(config['model'].getfloat('cond_scale', fallback=10.0))
	scroll.pack(side=tk.RIGHT, fill=tk.Y)
	label1.pack(side=tk.TOP, anchor=tk.N)
	entry1.pack(side=tk.TOP, anchor=tk.NW, ipadx=25, fill='x')
	check.pack()
	label2.pack()
	entry2.pack()
	label4.pack()
	entry4.pack()
	label5.pack()
	entry5.pack()
	label6.pack()
	entry6.pack()
	label7.pack()
	entry7.pack()
	label3.pack(side=tk.LEFT, padx=3, pady=8, anchor=tk.N)
	entry3.pack(side=tk.LEFT, padx=3, pady=8, anchor=tk.N)
	button2.pack(ipady=3, pady=3)
	button.pack(ipady=3, pady=3)
	button4.pack(side=tk.LEFT, anchor=tk.NW)
	button5.pack(side=tk.TOP, anchor=tk.NE)
	textframe.pack(anchor=tk.NE, fill='x')
	window.pack(side=tk.RIGHT, ipadx=10, anchor=tk.N)
	promptlabel.pack(side=tk.TOP, anchor=tk.S)
	promptframe.pack(side=tk.TOP, anchor=tk.S)
	butframe.pack(anchor=tk.N, ipady=3)
	picsframe.pack(side=tk.TOP)
	button3.pack(side=tk.LEFT, anchor=tk.SW)
	butframe2.pack(side=tk.BOTTOM, anchor=tk.S)
	window.after(5000, lambda: main2(checkval.get()))
	pics.mainloop()