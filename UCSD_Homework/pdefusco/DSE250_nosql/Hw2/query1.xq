let $text:="Et tu, Brute! Then fall, Caesar."
for $speaker in doc("j_caesar.xml")//PLAY/ACT/SCENE/SPEECH[LINE=$text]/SPEAKER
let $act := doc("j_caesar.xml")//PLAY/ACT[SCENE/SPEECH/LINE=$text]
return
<answer>
<who>{$speaker} </who>
<when> {$act/TITLE/text()} </when>
</answer>
