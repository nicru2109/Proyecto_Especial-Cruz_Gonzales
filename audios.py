## Configuracion de palabras SOLO SE GUARDAN UNA VEZ
from gtts import gTTS

p1 = gTTS(text = "Parpadeo doble", lang = 'es', slow = False)
p1.save("mp3//PP.mp3")

p2 = gTTS(text = "Parpadeo Izquierda", lang = 'es', slow = False)
p2.save("mp3//PI.mp3")

p3 = gTTS(text = "Parpadeo Derecha", lang = 'es', slow = False)
p3.save("mp3//PD.mp3")

p4 = gTTS(text = "Izquierda Parpadeo", lang = 'es', slow = False)
p4.save("mp3//IP.mp3")

p5 = gTTS(text = "Derecha Parpadeo", lang = 'es', slow = False)
p5.save("mp3//DP.mp3")

p6 = gTTS(text = "Frente", lang = 'es', slow = False)
p6.save("mp3//MF.mp3")

p7 = gTTS(text = "Mejilla Derecha", lang = 'es', slow = False)
p7.save("mp3//CD.mp3")

p8 = gTTS(text = "Mejilla Izquierda", lang = 'es', slow = False)
p8.save("mp3//CI.mp3")

p9 = gTTS(text = "Ambas mejillas", lang = 'es', slow = False)
p9.save("mp3//C.mp3")

p10 = gTTS(text = "Parpadeo sencillo", lang = 'es', slow = False)
p10.save("mp3//P.mp3")

p11 = gTTS(text = "Derecha", lang = 'es', slow = False)
p11.save("mp3//D.mp3")

p12 = gTTS(text = "Izquierda", lang = 'es', slow = False)
p12.save("mp3//I.mp3")

demo = gTTS(text = "Demostración", lang = 'es', slow = False)
demo.save("mp3//demo.mp3")

prep = gTTS(text = "Prepárate", lang = 'es', slow = False)
prep.save("mp3//prep.mp3")

go = gTTS(text = "Adelante", lang = 'es', slow = False)
go.save("mp3//go.mp3")