## Configuracion de palabras SOLO SE GUARDAN UNA VEZ
from gtts import gTTS

p1 = gTTS(text = "Parpadeo doble", lang = 'es', slow = False)
p1.save("mp3//p1.mp3")

p2 = gTTS(text = "Parpadeo Izquierda", lang = 'es', slow = False)
p2.save("mp3//p2.mp3")

p3 = gTTS(text = "Parpadeo Derecha", lang = 'es', slow = False)
p3.save("mp3//p3.mp3")

p4 = gTTS(text = "Izquierda Parpadeo", lang = 'es', slow = False)
p4.save("mp3//p4.mp3")

p5 = gTTS(text = "Derecha Parpadeo", lang = 'es', slow = False)
p5.save("mp3//p5.mp3")

p6 = gTTS(text = "Musculo Frontal", lang = 'es', slow = False)
p6.save("mp3//p6.mp3")

p7 = gTTS(text = "Cigomático Derecho", lang = 'es', slow = False)
p7.save("mp3//p7.mp3")

p8 = gTTS(text = "Cigomático Izquierdo", lang = 'es', slow = False)
p8.save("mp3//p8.mp3")

p9 = gTTS(text = "Ambos cigomáticos", lang = 'es', slow = False)
p9.save("mp3//p9.mp3")

p10 = gTTS(text = "Parpadeo solo", lang = 'es', slow = False)
p10.save("mp3//p10.mp3")

p11 = gTTS(text = "Ojo a la derecha", lang = 'es', slow = False)
p11.save("mp3//p11.mp3")

p12 = gTTS(text = "Ojo a la izquierda", lang = 'es', slow = False)
p12.save("mp3//p12.mp3")

demo = gTTS(text = "Demostración", lang = 'es', slow = False)
demo.save("mp3//demo.mp3")

prep = gTTS(text = "Prepárate", lang = 'es', slow = False)
prep.save("mp3//prep.mp3")

go = gTTS(text = "Adelante", lang = 'es', slow = False)
go.save("mp3//go.mp3")