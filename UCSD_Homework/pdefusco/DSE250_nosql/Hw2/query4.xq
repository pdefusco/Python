for $speaker in distinct-values(doc("j_caesar.xml")//PLAY/ACT/SCENE/SPEECH/SPEAKER)
where every $act in (doc("j_caesar.xml")//PLAY/ACT)
satisfies contains ($act, $speaker)
return
<result>
  <character>{$speaker}</character>
</result>
