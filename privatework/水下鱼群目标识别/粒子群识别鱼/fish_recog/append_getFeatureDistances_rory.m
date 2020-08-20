% This function takes two fish and a features index and returns the distance
% for that feature

% Fish1: Fish to be classified
% Fish2: Fish comparing against
% featureIndex: index of feature to use


function distance  = append_getFeatureDistances(feature1, feature2, featureIndex)
    if(featureIndex <= 24 )
        if(featureIndex == 1)
        %Head Histogram
            hist1 = feature1(1:243)';
            hist2 = feature2(1:243)';
        elseif(featureIndex == 2) 
        %Tail Histogram
            hist1 = feature1(244:486)';
            hist2 = feature2(244:486)';
        elseif(featureIndex == 3)
        %Top Histogram
            hist1 = feature1(487:729)';
            hist2 = feature2(487:729)';
        elseif(featureIndex == 4)
        %Bottom Histogram
            hist1 = feature1(730:972)';
            hist2 = feature2(730:972)';
        elseif(featureIndex == 5)
        %Whole Fish Histogram
            hist1 = feature1(973:1215)';
            hist2 = feature2(973:1215)';
        elseif(featureIndex == 6)
        %SectionedTailHist
            hist1 = feature1(1216:1458)';
            hist2 = feature2(1216:1458)';
        elseif(featureIndex == 7)
        %SectionedTailHist
            hist1 = feature1(1:81)';
            hist2 = feature2(1:81)';
        elseif(featureIndex == 8)
        %SectionedTailHist
            hist1 = feature1(82:162)';
            hist2 = feature2(82:162)';
        elseif(featureIndex == 9)
        %SectionedTailHist
            hist1 = feature1(163:243)';
            hist2 = feature2(163:243)';
        elseif(featureIndex == 10)
        %SectionedTailHist
            hist1 = feature1(244:324)';
            hist2 = feature2(244:324)';
        elseif(featureIndex == 11)
        %SectionedTailHist
            hist1 = feature1(325:405)';
            hist2 = feature2(325:405)';
        elseif(featureIndex == 12)
        %SectionedTailHist
            hist1 = feature1(406:486)';
            hist2 = feature2(406:486)';
        elseif(featureIndex == 13)
        %SectionedTailHist
            hist1 = feature1(487:567)';
            hist2 = feature2(487:567)';
        elseif(featureIndex == 14)
        %SectionedTailHist
            hist1 = feature1(568:684)';
            hist2 = feature2(568:684)';
        elseif(featureIndex == 15)
        %SectionedTailHist
            hist1 = feature1(649:729)';
            hist2 = feature2(649:729)';
        elseif(featureIndex == 16)
        %SectionedTailHist
            hist1 = feature1(730:810)';
            hist2 = feature2(730:810)';
        elseif(featureIndex == 17)
        %SectionedTailHist
            hist1 = feature1(811:891)';
            hist2 = feature2(811:891)';
        elseif(featureIndex == 18)
        %SectionedTailHist
            hist1 = feature1(892:972)';
            hist2 = feature2(892:972)';
        elseif(featureIndex == 19)
        %SectionedTailHist
            hist1 = feature1(973:1053)';
            hist2 = feature2(973:1053)';
        elseif(featureIndex == 20)
        %SectionedTailHist
            hist1 = feature1(1054:1134)';
            hist2 = feature2(1054:1134)';
        elseif(featureIndex == 21)
        %SectionedTailHist
            hist1 = feature1(1135:1215)';
            hist2 = feature2(1135:1215)';
        elseif(featureIndex == 22)
        %SectionedTailHist
            hist1 = feature1(1216:1296)';
            hist2 = feature2(1216:1296)';
        elseif(featureIndex == 23)
        %SectionedTailHist
            hist1 = feature1(1297:1377)';
            hist2 = feature2(1297:1377)';
        elseif(featureIndex == 24)
        %SectionedTailHist
            hist1 = feature1(1378:1458)';
            hist2 = feature2(1378:1458)';
        
        end
        distance = append_compareChisquare(hist1,hist2);
    else
        distance = abs(feature1(featureIndex-24)-feature2(featureIndex-24));
    end
end