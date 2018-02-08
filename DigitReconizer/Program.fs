open System.IO

type Observation = { Label:string; Pixels:int[] }

let toObservation (csvData:string) =
    let colums = csvData.Split(',')
    let label = colums.[0]
    let pixels = colums.[1..] |> Array.map int
    { Label = label; Pixels = pixels}

let reader path =
    let data = File.ReadAllLines path
    data.[1..]
    |> Array.map toObservation

let trainingPath = @"/Users/malte/Documents/Schantz/MachineLearning/trainingsample.csv"
let trainingData = reader trainingPath
