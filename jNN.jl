module jNN
export NN

importall Layers

type NN
	layers::Array{Any}
	
	addHidden::Function
	addOutput::Function
	propagate::Function
	train::Function

	function NN(n::Int64)
		this = new([Input(n)])

		function addHidden(n::Int64)
			push!(this.layers, Hidden(this.layers[end].size, n))
		end

		function addOutput(n::Int64)
			push!(this.layers, Output(this.layers[end].size, n))
		end

		function propagate(input::Array{Float64, 1})
			for layer in this.layers
				input = layer.propagate(input)
			end
			return input
		end

		function train(inputs, targets, iterations = 50000, learningRate = 0.1)
			for it = 0:iterations
				for i = 1:length(inputs)
					input = deepcopy(inputs[i])
					target = deepcopy(targets[i])

					this.propagate(input)

					for i = 0:(length(this.layers) - 1)
						target = this.layers[end - i].train(target, learningRate)
					end

					for layer in this.layers
						layer.update()
					end
				end
			end
		end

		this.addHidden = addHidden
		this.addOutput = addOutput
		this.propagate = propagate
		this.train = train

		return this
	end
end

end #module
