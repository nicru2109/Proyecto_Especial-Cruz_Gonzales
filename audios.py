## Configuracion de palabras SOLO SE GUARDAN UNA VEZ
from gtts import gTTS

p1 = gTTS(text = "Hola", lang = 'es', slow = False)
p1.save("mp3//p1.mp3")

p2 = gTTS(text = "Sí", lang = 'es', slow = False)
p2.save("mp3//p2.mp3")

p3 = gTTS(text = "No", lang = 'es', slow = False)
p3.save("mp3//p3.mp3")

p4 = gTTS(text = "No estoy de acuerdo", lang = 'es', slow = False)
p4.save("mp3//p4.mp3")

p5 = gTTS(text = "Me duele", lang = 'es', slow = False)
p5.save("mp3//p5.mp3")

p6 = gTTS(text = "Adiós", lang = 'es', slow = False)
p6.save("mp3//p6.mp3")

demo = gTTS(text = "Demostración", lang = 'es', slow = False)
demo.save("mp3//demo.mp3")

prep = gTTS(text = "Prepárate", lang = 'es', slow = False)
prep.save("mp3//prep.mp3")

go = gTTS(text = "Adelante", lang = 'es', slow = False)
go.save("mp3//go.mp3")