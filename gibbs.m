function v = gibbs(v1,h_bias,v_bias,W,steps)

for i = 1:steps
    hidden_p = sigmoid(v1*W+h_bias);   
    %h = zeros(size(hidden_p));                           
    %h(hidden_p >= rand(size(hidden_p))) = 1;   
    
    visible_p = sigmoid(hidden_p*W'+v_bias);
    v = zeros(size(visible_p));                           
    v(visible_p >=rand(size(visible_p))) = 1;
    
    v1 = v;
end

end    