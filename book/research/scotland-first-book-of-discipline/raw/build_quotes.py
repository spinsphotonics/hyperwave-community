import json

quotes = []

def add(qid, doc, page, text, before, after, fields, epigraph=False):
    quotes.append({
        "quote_id": qid,
        "document_id": doc,
        "page": f"p. {page}",
        "text": text,
        "before": before,
        "after": after,
        "language": "en",
        "translation": "",
        "translation_source": "",
        "fields": fields,
        "epigraph_candidate": epigraph,
    })

add("scotland-fbd-q001","laing1848-fbd","183",
    "THE PREFACE TO THE BUKE OF DISCIPLLNE",
    "other editions/MSS, are included in the OCR as printed, interleaved with the main text) WORD COUNT: 27539 [[p. 183]]",
    "/ To THE Great Counsall of Scotland now admitted TO [the] Regiment, by the Providence of God, and BY THE",
    ["F1","F5"])

add("scotland-fbd-q002","laing1848-fbd","183",
    "Frome youre Honouris we receaved a charge, daittit at Edinburgh, xxix of Aprile, in tlio yeir of God",
    "and Peace from God the Father of ourb Lord Jesus Christ, with the perpetuall encrease of the Holye Spirite.",
    "J°' V thre scoir yeiris, requyring and commanding us, in the name of the THE FIRST AND SECOND BOOKE",
    ["F2"])

add("scotland-fbd-q003","laing1848-fbd","184",
    "to committ to Avriting, and in a Buke to deliver unto your Wisdomes oure jugementis tuiching the Reformatioun of Religioun",
    "been omitted by the printer.\" It is to [[p. 184]] Eternall God, as we will ansuer in his presence,",
    ", quliilk heirtofore in this Realme, (as in utheris,) hes hene utterlie corrupted. Upone the recept quhairof, sa mony of",
    ["F6"])

add("scotland-fbd-q004","laing1848-fbd","184",
    "quliilk heirtofore in this Realme, (as in utheris,) hes hene utterlie corrupted",
    "committ to Avriting, and in a Buke to deliver unto your Wisdomes oure jugementis tuiching the Reformatioun of Religioun,",
    ". Upone the recept quhairof, sa mony of us as wer in this Toune, did convene, and in unitie of",
    ["F7"])

add("scotland-fbd-q005","laing1848-fbd","185",
    "Seeing that Christ Jesus is he quhoni God the Father hes commandit onlie to he herd, and followed of his scheip, we urge it necessarie, that his Evangell^ be trewlie and openlie preached in everie Kirk and Assemblie of this Realme",
    "In edit. 1621, \" pleasure and.\" in Vautr. edit, and edit. 1621. [[p. 185]] The First Head, of Doctrine.i",
    "; and that all doctrine repugnyng^ to the same he utterlie suppres- sed^ as damnahill to mannis salvatioun. The",
    ["F8"])

add("scotland-fbd-q006","laing1848-editorial-note","181",
    "It had been formed by John Knox, partly in imitation of the reformed Churches of Germany, partly of that he had seen in Geneva",
    "Book of Discipline. At the conclusion he adds, — \" This was the Policy desu-ed to be ratified :",
    "; whence he took that device of annuall Deacons for collecting and dispensing the Church rents, whereof in the",
    ["F9"])

add("scotland-fbd-q007","laing1848-fbd","209",
    "oif necessitie it is that your Hoiiouris be most cairfull for the virtuous educatioun, and godlie up- bringing of the youth of this Realme",
    "illuminat men miraculuslie, suddanlie changeing thame, as that he did his Apostlis and utheris in the Primitive Churche :",
    ", yf eathir ye now thirst un- feanedlie [for] the advancement of Christis glorie, or yit desire the continewance of",
    ["F6","M1"], True)

add("scotland-fbd-q008","laing1848-fbd","209",
    "everie severall Churche have a Scholmaister^ appointed, suche a one as is able, at least, to teache Gramraer and the Latine toung",
    "to us, to wit, the Churche and Spouse of the Lord Jesus. Off necessitie thairfore we judge it, that",
    ", yf the Toun be of any reputatioun. Yf it be Upaland, whaire the people convene to doctrine bot once",
    ["C1","C2","A2"])

add("scotland-fbd-q009","laing1848-fbd","210",
    "in everie notable toun, and especiallie in the toun of the Superintendent, [there] be erected a Colledge, in whiche the Artis, at least Logick and Rethorick, togidder with the Tongues, be read",
    "in the Booke of our Common Ordour, callit the Ordour of Geneva.^ And farther, we think it expedient, that",
    "be sufficient Maisteris, for whome honest stipendis must be appointed : as also provisioun for those that be poore,",
    ["C2","C3"])

add("scotland-fbd-q010","laing1848-fbd","211",
    "The riche and potent may not be permitted to suffer thair children to spend thair youth in vane idilnes, as heirtofore thei have done",
    "especialhe in thair youth-hcade ; but all must be compelled to bring- up thair children in leamyng and virtue.",
    ". But thei must be exhorted, and by the censure of the Churche compelled to dedicat thair sones, by goode",
    ["M2","M3"])

add("scotland-fbd-q011","laing1848-fbd","211",
    "The children of the poore must be supported and sustenit on the charge of the Churche, till tryell be tackin, whethir the spirit of docilitie be fund in them or not",
    "Churche and to the Common-wealth ; and that thei must do of thair awin expensses, becaus thei ar able.",
    ". Yf thei be fund apt to letteris and learnyng, then may thei not (we meane, neathir the sonis of",
    ["A2","M2","L6"])

add("scotland-fbd-q012","laing1848-fbd","211",
    "the Ministeris and Elderis, with the best learned in everie toun, shall everie quarter tak examinatioun",
    "men be appointit to visit all Schollis for the tiyell of thair exercise, proffit, and continewance ; to Avit,",
    "^ how the youth hath proffitted. A certane tyme must be appointed to Reiding, and to learn- ing of the",
    ["C7"])

add("scotland-fbd-q013","laing1848-fbd","212",
    "Two yearis we think more then sufficient to learne to read perfitelie, to answer to the Catechisme",
    "And thairfore, these principallis aught and must be learned in the youth-heid. II. The tymes appointed to everie Course.",
    ", and to have some en- tresse in the first rudimontis of Grammar ; to the full accom- plischement whairof,",
    ["C1","C2"])

add("scotland-fbd-q014","laing1848-fbd","212",
    "the rest, till the aige of twenty- foure yearis to be spent in that studye",
    "most, sufficient. To the Artis, to wit, Logick and Rethorick, and to the Greik toung, foure yeiris ; and",
    ", whairin the learnar wald proffit the Churche or Commoun-wealth, be it in the Lawis, or Physick or Divinitie :",
    ["C1"])

add("scotland-fbd-q015","laing1848-fbd","213",
    "thair be three Uni- versities in this whole Reahne, establischeit in the Tounis accustumed",
    "Universiteis. The Grammar Schollis and of the Toiingis being erectit as we have said, nixt we think it necessarie",
    ".i The first in Sanctandrois,*^ the secound in Glasgow,^ and the thrid in Abirdene.^ And in the first Universitie and",
    ["C3","F4"])

add("scotland-fbd-q016","laing1848-fbd","214",
    "shall be fund sufficientlie instructit in thir aforesaid sciences, shall be Laureat and Gradiiat in Philosophie",
    "who shall compleit his coiirse in a yeare. And wha efter thir thre yearis, by tryell and exami- natioun,",
    ". In the fourt classe, shall be ane Reidar of Medicine, Avho shall com- pleit his course in five years",
    ["C8"])

add("scotland-fbd-q017","laing1848-fbd","214",
    "the Reidar of the Hebreu shall inter- preit ane booke of Moses, the",
    "Greek toung, wha sail compleit the grammeris thairof in half ane yeare, l and the remanent of the yeare,",
    "2 Propheitis, or the Psalmes ; sa that his course and classe shall continew ane yeare. The Rei- dar",
    ["C4","C5"])

add("scotland-fbd-q018","laing1848-fbd","214",
    "The Rei- dar of the Greek shall interpreit some booke of Plato, to- gidder with some place of the New Testament",
    "of Moses, the 2 Propheitis, or the Psalmes ; sa that his course and classe shall continew ane yeare.",
    ". And in the secound classe, shalbe tuo Reideris in Divinitie, that ane in the New Testament, that uthir in",
    ["C4","C5"])

add("scotland-fbd-q019","laing1848-fbd","214",
    "nane be admittit unto the first Colledge, and to be Suppostis of the Universitie, onles he have frome the Maister of the Scheie, and the Minister of the toun whair he was instructed in the toungis, ane testimoniall of his learnyng, docilitie, aige, and parentage",
    "whiche tyme, who sail be fund by examinatioun sufficient shall be graduat in Divinitie. Item, We think expedient that",
    "; and likewayis ' In edit. 1621, \" in three monetlis.\" - In edit. 1722, '• or of the.\"",
    ["A2","A3","A4"])

add("scotland-fbd-q020","laing1848-fbd","218",
    "in Sanctandrois, seventie-tua bursaris ; in Glasgou, fourtye-eyght bursaris ; in Abirdene, fourty-eyght",
    "everieUniversitie,thair he twenty-four hursaris,! divided equalie in all the classes and seigeis, as is above exprimit : that is,",
    "; to be sustened onlie in meit upon the chargeis of the Colledge ; and be admitted at the",
    ["L7","L6"])

add("scotland-fbd-q021","laing1848-fbd","219",
    "everie Erlis sone, at his entre to the Universitie, shall gif fourtye schillingis",
    "We have thocht gude for building and uphald of the placis, ane general collect be maid ; and that",
    ", and sicklike at everie graduatioun, 40 schillingis. Item, Everie Lordis sone sicklike at ilk tyme, 80 schillingis ; ilk",
    ["L6"])

add("scotland-fbd-q022","laing1848-fbd","216",
    "shall dalie hearkin the dyet comptis ; adjoynyng to him oulklie ane of the Readeris or Regentis, above whome he shall [take] attendence upoun thair diligence",
    "the haill rentis of the Colledge, and distribute the same according to the erec- tioun of the Colledge, and",
    ", alsweill in thair reading, as exercitioun''' of the 3^outh in the mater taught ; upoun the polecye and uphold",
    ["T2","T3"])

add("scotland-fbd-q023","laing1848-fbd","216",
    "shall bald ane oulklie 8 conventioun with the haill memberis of the Colledge",
    "in the mater taught ; upoun the polecye and uphold of the place ; and for punischement of crymes,",
    ". He shall be comptabile yearlie to the Superintendent, Rectour, and rest of the Principallis convened, about the first of",
    ["T3","L4"])

add("scotland-fbd-q024","laing1848-fbd","216",
    "ane Steward, ane Cooke, ane Gardnar, ane Portar, wha shall be subject to discipline of the Principale, as the rest",
    "with singill ee,^ but respect to feid or favour. Item, In everie Colledge, we think neidfull at the least",
    ". Itevt, That everie Universitie have ane Beddale subject to Greek ; but it is probable that a Profes- iii",
    ["L4","T1"])

add("scotland-fbd-q025","laing1848-fbd","212",
    "the learnar most be removed to serve the Churche or Commoun-wealth, unless he be fund a necessarie Reidare in the same Colledge or Universitie",
    "it in the Lawis, or Physick or Divinitie : Wliiche tyme of twenty-foure yearis being spent in the schollis,",
    ". Yf God shall move your heartis to establische and execut this Ordour, and put these thingis in practise, your",
    ["S1","S4","C8"])

add("scotland-fbd-q026","laing1848-fbd","257",
    "Act of Secreit Counsall, xxvii Januarii, Anno &c., lx°.i We, quhilkis hes subscryvit thir Presentis, haveand avysit with the Articles heirin specifeit",
    "Posteriteis following. Amen. So be it. By your Honouris Most humble Servitouris, etc. Frome Edinburgh, The 20of Mciij 1560.",
    ", as is abone mentionat fra the begynning. of this Book, thinkis the samin good, and con- forme to Goddis",
    ["F2","F8"])

add("scotland-fbd-q027","laing1848-fbd","257",
    "That is, the 27th (in edit. 1621, the",
    "Bischoppis, Abbotis, Priouris, and otheris Prelattis and beneficit men, quhilkis ellis hes adjonit thame to us, bruik the 1",
    "did not begin at that time till the 2oth 17th) of January 1560-1; as the year of March. VOL.",
    ["F2"])

add("scotland-fbd-q028","laing1848-fbd","215",
    "triall to be tanei be certan Exaniinatouris, deput be the Rec- toiir and Principallis of the same, and yf he be fund sufficient- lie instructit in Dialectick",
    "likewayis ' In edit. 1621, \" in three monetlis.\" - In edit. 1722, '• or of the.\" [[p. 215]]",
    ",^ he shall incontinent, that same yeare, be promoted to the classe of Mathematicque. Item, That nane be admittit to",
    ["A4","C7"])

with open("/home/user/hyperwave-community/book/research/scotland-first-book-of-discipline/quotes.jsonl", "w", encoding="utf-8") as f:
    for q in quotes:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print("wrote", len(quotes), "quotes")
