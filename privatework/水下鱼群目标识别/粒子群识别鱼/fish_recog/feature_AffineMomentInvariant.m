function [ feature ] = feature_AffineMomentInvariant( mask_image )
mask_image = double(mask_image);
f=append_ami_readinv('append_ami_afinv.txt'); % reading of the invariants from a proper file 
mm=append_ami_cm(mask_image,12); % moment computation 
feature=append_ami_cafmi(f,mm); % invariant evaluation 

end

