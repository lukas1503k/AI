package main

//Lukas Phangureh
//Using the library github.com/patrikeh/go-deep
//A training of a neural network for three input XOR, using the sample code for the

import (
	"fmt"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func main() {
	//The truth table for 3 input XOR
	var sampleData = training.Examples{
		{[]float64{0, 0, 0}, []float64{0}},
		{[]float64{0, 0, 1}, []float64{1}},
		{[]float64{0, 1, 0}, []float64{1}},
		{[]float64{0, 1, 1}, []float64{0}},
		{[]float64{1, 0, 0}, []float64{1}},
		{[]float64{1, 0, 1}, []float64{0}},
		{[]float64{1, 1, 0}, []float64{0}},
		{[]float64{1, 1, 1}, []float64{1}},
	}

	n := deep.NewNeural(&deep.Config{
		/* Input dimensionality */
		Inputs: 3,
		Layout: []int{3, 3, 1}, // defines a neural network with 3 inputs, a hidden layer with 4 weights and a single output
		//Suggested default values
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.NewNormal(1.0, 0.0),
		Bias:       true, //enable bias
	})
	optimizer := training.NewSGD(0.05, 0.1, 1e-6, true)
	trainer := training.NewTrainer(optimizer, 50)

	training, heldout := sampleData.Split(0.5)
	trainer.Train(n, training, heldout, 10000)

	//The predictions for each

	fmt.Println(sampleData[0].Input, "=>", n.Predict(sampleData[0].Input))
	fmt.Println(sampleData[1].Input, "=>", n.Predict(sampleData[1].Input))
	fmt.Println(sampleData[2].Input, "=>", n.Predict(sampleData[2].Input))
	fmt.Println(sampleData[3].Input, "=>", n.Predict(sampleData[3].Input))
	fmt.Println(sampleData[4].Input, "=>", n.Predict(sampleData[4].Input))
	fmt.Println(sampleData[5].Input, "=>", n.Predict(sampleData[5].Input))
	fmt.Println(sampleData[6].Input, "=>", n.Predict(sampleData[6].Input))
	fmt.Println(sampleData[7].Input, "=>", n.Predict(sampleData[7].Input))

}
