function v = gibbs(v1,h_bias,v_bias,W,steps)

for i = 1:steps
    hidden_p = sigmoid(v1*W+h_bias);   
    h = zeros(size(hidden_p));                           
    h(hidden_p >= rand(size(hidden_p))) = 1;   
    
    visible_p = sigmoid(h*W'+v_bias);
    v = zeros(size(visible_p));                           
    
    condition = ((v1 == 1) | ((v1 == 0.5) & visible_p >= rand(size(visible_p))));
    v(condition) = 1;
    v1 = v;
end

end    