function video_feat = process_box_feat3(box_feat, video_info, stride)

nVideo = size(video_info, 1);
video_feat = zeros(size(box_feat, 1), nVideo);
for n = 1:nVideo
    feature_set = box_feat(:, video_info(n, 1):video_info(n, 2)-1);    
    feature_set = feature_set( :, 1:stride:end);
    video_feat(:, n) = mean(feature_set, 2); % avg pooling
    %video_feat(:, n) = feature_set(:, 1);
end

%%% normalize train and test features
sum_val = sqrt(sum(video_feat.^2));
for n = 1:size(video_feat, 1)
    video_feat(n, :) = video_feat(n, :)./sum_val;
end