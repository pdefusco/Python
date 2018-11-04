for $s in distinct-values(doc("j_caesar.xml")//PLAY/ACT/SCENE/SPEECH/SPEAKER)
let $c := count(doc("j_caesar.xml")//PLAY/ACT/SCENE/SPEECH[SPEAKER=$s]/LINE)
let $act := (doc("j_caesar.xml")//PLAY/ACT[SCENE/SPEECH/SPEAKER=$s])
where $c>0
return
<result>
<answer>
<who>{ $s }</who>
<when>{$act/TITLE/text()}</when>
</answer>
</result>
