import jNN.NN

inputs = ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])
labels = ([0.0], [1.0], [1.0], [0.0])

brain = NN(2)
brain.addHidden(2)
brain.addOutput(1)

brain.train(inputs, labels)

println("----------+---------------------")
println("   input  |       output")
println("----------+---------------------")
for input in inputs
    output = brain.propagate(input)
    println("$input | $output")
end
