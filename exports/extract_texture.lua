
                function extractTexture()
                    local textureId = 2180015
                    local texturePath = "./exports/crackers_visual.png"
                    
                    -- Try to save the texture
                    local result = sim.saveTexture(textureId, texturePath, 0, 0, 0, 0, 0, 0, 0, 0)
                    
                    if result == 0 then
                        print("Texture extracted successfully to: " .. texturePath)
                        return true
                    else
                        print("Failed to extract texture. Error code: " .. result)
                        return false
                    end
                end
                
                extractTexture()
                