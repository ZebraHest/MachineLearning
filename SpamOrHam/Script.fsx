open System.IO

type DocType =
    | Ham
    | Spam
   
let parseDocType (label:string) =
    match label with
    | "ham" -> Ham
    | "spam" -> Spam
    | _ -> failwith "Unknown label"

let parseLine(line:string) =
    let split = line.Split('\t')
    let label = split.[0] |> parseDocType
    let message = split.[1]
    (label, message)


let path = @"/Users/malte/Documents/Schantz/MachineLearning/Data/SMSSpamCollection.txt"

let dataset =
    File.ReadAllLines path
    |> Array.map parseLine

let spamWithFREE =
    dataset
    |> Array.filter (fun (docType,_) -> docType = Spam)
    |> Array.filter (fun (_,sms) -> sms.Contains("FREE"))
    |> Array.length

let hamWithFREE =
    dataset
    |> Array.filter (fun (docType,_) -> docType = Ham)
    |> Array.filter (fun (_,sms) -> sms.Contains("FREE"))
    |> Array.length

open System.Text.RegularExpressions

let matchWords = Regex(@"\w+")

let wordTokenizer (text:string) =
    text.ToLowerInvariant()
    |> matchWords.Matches
    |> Seq.cast<Match>
    |> Seq.map (fun m -> m.Value)
    |> Set.ofSeq

#load "NaiveBayes.fs"
open NaiveBayes.Classifier

let validation, training = dataset.[..999], dataset.[1000..]

let txtClassifier = train training wordTokenizer (["txt"] |> set)

validation
|> Seq.averageBy (fun(docType,sms) ->
    if docType = txtClassifier sms then 1.0 else 0.0)
|> printfn "Based on 'txt', correctly classified: %.3f"

let vocabulary (tokenizer:Tokenizer) (corpus:string seq)=
    corpus
    |> Seq.map tokenizer
    |> Set.unionMany

let allTokens =
    training
    |> Seq.map snd
    |> vocabulary wordTokenizer

let fullClassifier = train training wordTokenizer allTokens

validation
|> Seq.averageBy (fun(docType,sms) ->
    if docType = fullClassifier sms then 1.0 else 0.0)
|> printfn "Based on all tokens, correctly classified: %.3f"

let evaluate (tokenizer:Tokenizer) (tokens:Token Set) =
    let classifier = train training tokenizer tokens
    validation
    |> Seq.averageBy (fun(docType,sms) ->
        if docType = classifier sms then 1.0 else 0.0)
    |> printfn "Correctly classified: %.3f"

let casedTokenizer (text:string) =
    text
    |> matchWords.Matches
    |> Seq.cast<Match>
    |> Seq.map (fun m -> m.Value)
    |> Set.ofSeq

let casedTokens =
    training
    |> Seq.map snd
    |> vocabulary casedTokenizer

evaluate casedTokenizer casedTokens

let top n (tokenizer:Tokenizer) (docs:string []) =
    let tokenized = docs |> Array.map tokenizer
    let tokens = tokenized |> Set.unionMany
    tokens
    |> Seq.sortBy (fun t -> - countIn tokenized t)
    |> Seq.take n
    |> Set.ofSeq

let ham, spam =
    let rawHam, rawSpam =
        training
        |> Array.partition (fun (lbl,_) -> lbl=Ham)
    rawHam |> Array.map snd,
    rawSpam |> Array.map snd

let hamCount = ham |> vocabulary casedTokenizer |> Set.count
let spamCount = spam |> vocabulary casedTokenizer |> Set.count

let topHam = ham |> top (hamCount /10) casedTokenizer
let topSpam = spam |> top (spamCount /10) casedTokenizer

let topTokens = Set.union topHam topSpam

evaluate casedTokenizer topTokens

ham |> top 20 casedTokenizer |> Seq.iter (printf "%s, ")
spam |> top 20 casedTokenizer |> Seq.iter (printf "%s, ")

let commenTokens = Set.intersect topHam topSpam
let specificTokens = Set.difference topTokens commenTokens
evaluate casedTokenizer specificTokens

let rareTokens n (tokenizer:Tokenizer) (docs:string []) =
    let tokenized = docs |> Array.map tokenizer
    let tokens = tokenized |> Set.unionMany
    tokens
    |> Seq.sortBy (fun t -> countIn tokenized t)
    |> Seq.take n
    |> Set.ofSeq

ham |> rareTokens 20 casedTokenizer |> Seq.iter (printf "%s, ")
spam |> rareTokens 20 casedTokenizer |> Seq.iter (printf "%s, ")

let phoneWords = Regex(@"[7-9]\d{9}")
let phone (text:string) = 
    match(phoneWords.IsMatch text) with
    | true -> "__PHONE__"
    | false -> text

let txtCode = Regex(@"\b\d{5}\b")
let txt (text:string) = 
    match(txtCode.IsMatch text) with
    | true -> "__TXT__"
    | false -> text

