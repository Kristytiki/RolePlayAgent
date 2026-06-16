# Persona portraits — sources & licensing

Portraits live in `RolePlayChatbotUI/public/portraits/<persona_id>.jpg`
and are served by Vite at `/portraits/<id>.jpg`. Each is downloaded from
Wikipedia / Wikimedia Commons under terms compatible with display in this
project (predominantly public domain; a few are CC-BY-SA).

| persona | file | source | rights |
|---|---|---|---|
| Mr. Darcy | `mr_darcy.jpg` | https://commons.wikimedia.org/wiki/File:PrideandPrejudiceCH3detail.jpg | C.E. Brock 1895 illustration — public domain |
| Elizabeth Bennet | `elizabeth_bennet.jpg` | https://commons.wikimedia.org/wiki/File:Elisabeth_Bennet_(d%C3%A9tail).jpg | C.E. Brock 1895 illustration — public domain |
| Heathcliff | `heathcliff.jpg` | https://commons.wikimedia.org/wiki/File:Houghton_Lowell_1238.5_(A)_-_Wuthering_Heights,_1847.jpg | 1847 first-edition title page — public domain |
| Sherlock Holmes | `sherlock_holmes.jpg` | https://commons.wikimedia.org/wiki/File:Sherlock_Holmes_Portrait_Paget.jpg | Sidney Paget 1891 illustration — public domain |
| Jay Gatsby | `jay_gatsby.jpg` | https://commons.wikimedia.org/wiki/File:Warner_Baxter_as_Jay_Gatsby_in_The_Great_Gatsby_(1926)_Retouched_Cropped.jpg | Warner Baxter 1926 publicity still — public domain |
| Hermione Granger | `hermione_granger.jpg` | https://commons.wikimedia.org/wiki/File:Harry_Potter_Book_1,_1st_American_ed._without_dust_jacket.JPG | Harry Potter Vol. 1 spine, 1st US edition (no PD canon portrait of Hermione exists) |
| Anna Karenina | `anna_karenina.jpg` | https://commons.wikimedia.org/wiki/File:AnnaKareninaTitle.jpg | First Russian-language edition title page (1877) — public domain |
| Atticus Finch | `atticus_finch.jpg` | https://commons.wikimedia.org/wiki/File:Gregory_Peck_Atticus_Publicity_Photo.jpg | Universal Studios 1962 publicity photo — public domain (no copyright notice on original) |
| Scarlett O'Hara | `scarlett_ohara.jpg` | https://commons.wikimedia.org/wiki/File:Vivien_Leigh_Gone_Wind_Restored.jpg | MGM 1939 publicity still, restored crop — public domain (no renewal) |

If you want fully self-hosted images, regenerate via `scripts/`-style
download (see `RolePlayChatbotUI/public/portraits/` paths in the diff
that introduced this directory).
