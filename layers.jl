module Layers
export Input, Hidden, Output # types

logistic(x) = 1 ./ (1 + exp(-x))
d_logistic(x) = logistic(x) .* (1 - logistic(x))

type Input
    size::Int64

    propagate::Function
    train::Function
    update::Function

    function Input(n::Int64) 
        this = new(n)

        propagate(input::Array{Float64, 1}) = return input
        train(last_delta, learningRate) = return last_delta
        update() = return

        this.propagate = propagate
        this.train = train
        this.update = update

        return this
    end
end

type Hidden
    inputs::Array{Float64, 1}
    activations::Array{Float64, 1}
    outputs::Array{Float64, 1}

    size::Int64

    weights::Array{Float64, 2}
    bias::Array{Float64, 1}

    weight_gradients::Array{Float64, 2}
    bias_gradients::Array{Float64, 1}

    propagate::Function
    train::Function
    update::Function

    function Hidden(n_in::Int64, n_out::Int64)
        this = new(zeros(n_in), zeros(n_out), zeros(n_out), n_out, rand(n_out, n_in), zeros(n_out), zeros(n_out, n_in), zeros(n_out))

        function propagate(input::Array{Float64, 1})
            this.inputs = input
            this.activations = (this.weights * this.inputs) + this.bias
            this.outputs = logistic(this.activations)
        end

        function train(last_delta::Array{Float64, 1}, learningRate::Float64)
            delta = last_delta .* d_logistic(this.activations)
            this.weight_gradients = learningRate .* (delta * this.inputs.')
            this.bias_gradients = learningRate .* delta
            next_delta = this.weights.' * delta
        end

        function update()
            this.weights -= this.weight_gradients
            this.bias -= this.bias_gradients
        end

        this.propagate = propagate
        this.train = train
        this.update = update

        return this
    end 
end

type Output
    inputs::Array{Float64, 1}
    activations::Array{Float64, 1}
    outputs::Array{Float64, 1}

    size::Int64

    weights::Array{Float64, 2}
    bias::Array{Float64, 1}

    weight_gradients::Array{Float64, 2}
    bias_gradients::Array{Float64, 1}

    propagate::Function
    train::Function
    update::Function

    function Output(n_in::Int64, n_out::Int64)
        this = new(zeros(n_in), zeros(n_out), zeros(n_out), n_out, rand(n_out, n_in), zeros(n_out), zeros(n_out, n_in), zeros(n_out))

        function propagate(input::Array{Float64, 1})
            this.inputs = input
            this.activations = (this.weights * this.inputs) + this.bias
            this.outputs = logistic(this.activations)
        end

        function train(target::Array{Float64, 1}, learningRate::Float64)
            delta = this.outputs - target
            this.weight_gradients = learningRate .* (delta * this.inputs.')
            this.bias_gradients = learningRate .* delta
            next_delta = this.weights.' * delta
        end

        function update()
            this.weights -= this.weight_gradients
            this.bias -= this.bias_gradients
        end

        this.propagate = propagate
        this.train = train
        this.update = update

        return this
    end 
end

end #module	

