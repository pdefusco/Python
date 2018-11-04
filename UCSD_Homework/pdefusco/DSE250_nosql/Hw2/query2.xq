for $s in distinct-values(doc("j_caesar.xml")//PLAY/ACT/SCENE/SPEECH/SPEAKER)
  let $c := count(doc("j_caesar.xml")//PLAY/ACT/SCENE/SPEECH[SPEAKER=$s]/LINE)
  where $c>0
    return
    <speakers>
    <character>{ $s }</character>
    </speakers>
