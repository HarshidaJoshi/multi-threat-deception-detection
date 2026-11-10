"""Revert paper.tex to pre-50-reference bibliography."""
from pathlib import Path
import re

tex_path = Path("docs/sicacs2026_llncs/paper.tex")
text = tex_path.read_text(encoding="utf-8")

rev = {
    "ref21": "mirsky2021",
    "ref09": "soundarya2026",
    "ref10": "bhat2026",
    "ref25": "alghamdi2023",
    "ref36": "alshuwaier2025",
    "ref46": "taneja2025",
    "ref01": "moyo2026",
    "ref03": "kumar2026",
    "ref05": "deepa2026",
    "ref13": "javiya2025",
    "ref11": "soudy2024",
    "ref07": "magesh2025",
    "ref08": "castro2026",
    "ref14": "mansoor2025",
    "ref15": "momin2025",
    "ref22": "tsigos2024",
    "ref29": "nimah2025",
    "ref32": "rama2025",
    "ref37": "kumarc2025",
    "ref43": "alqadi2025",
    "ref04": "chaleh2025",
    "ref31": "sneha2026",
    "ref50": "men2024",
    "ref34": "bezzaoui2025",
    "ref44": "naude2023",
    "ref47": "franceschini2026",
    "ref48": "asiama2026",
    "ref24": "abbas2024",
    "ref39": "lv2025",
    "ref06": "omar2024",
    "ref19": "duan2025",
    "ref38": "gk2025",
    "ref33": "almandouh2024",
}

body, _, _ = text.partition(r"\begin{thebibliography}")
if "Complementary corpus papers used for breadth" in body:
    body = re.sub(
        r" Purely sociological scam studies appear where they clarify threat context, "
        r"even when they do not introduce a classifier\. Complementary corpus papers "
        r"used for breadth---.*?---are included in the full reference list\.",
        " Purely sociological scam studies appear where they clarify threat context, "
        "even when they do not introduce a classifier.",
        body,
        count=1,
        flags=re.S,
    )


def repl_cite(m):
    parts = [p.strip() for p in m.group(1).split(",")]
    return r"\cite{" + ",".join(rev.get(p, p) for p in parts) + "}"


body = re.sub(r"\\cite\{([^}]*)\}", repl_cite, body)

old_bib = r"""\begin{thebibliography}{99}

\bibitem{mirsky2021}
Mirsky, Y., Lee, W.: The creation and detection of deepfakes: a survey. ACM Computing Surveys \textbf{54}(1), 1--41 (2021). \doi{10.1145/3425780}

\bibitem{soundarya2026}
Soundarya, B.C., Gururaj, H.L.: Deepfake detection: critical review of state-of-the-art approaches and future perspectives. Discover Applied Sciences \textbf{8}(2) (2026). \doi{10.1007/s42452-025-08174-9}

\bibitem{bhat2026}
Bhat, N.A., Giri, K.J.: DeepFake detection in images and videos: a survey on models, datasets, and evaluation metrics. Multimedia Tools and Applications \textbf{85}(3) (2026). \doi{10.1007/s11042-026-21365-9}

\bibitem{alghamdi2023}
Alghamdi, J., Luo, S., Lin, Y.: A comprehensive survey on machine learning approaches for fake news detection. Multimedia Tools and Applications \textbf{83}(17), 51009--51067 (2023). \doi{10.1007/s11042-023-17470-8}

\bibitem{alshuwaier2025}
Alshuwaier, F.A., Alsulaiman, F.A.: Fake news detection using machine learning and deep learning algorithms: a comprehensive review and future perspectives. Computers \textbf{14}(9), 394 (2025). \doi{10.3390/computers14090394}

\bibitem{taneja2025}
Taneja, K., Vashishtha, J., Ratnoo, S.: Fraud-BERT: transformer based context aware online recruitment fraud detection. Discover Computing \textbf{28}(1) (2025). \doi{10.1007/s10791-025-09502-8}

\bibitem{moyo2026}
Moyo, B.V., Tuyikeze, T., Matsebula, F., et al.: An AI-driven conceptual framework for detecting fake news and deepfake content: a systematic review. Frontiers in Artificial Intelligence \textbf{9} (2026). \doi{10.3389/frai.2026.1737790}

\bibitem{kumar2026}
Kumar, M., Kumar, A., Yadav, V., et al.: A hybrid spatial--frequency attention-based algorithm using efficientnet for robust and interpretable deepfake detection. Scientific Reports \textbf{16}(1) (2026). \doi{10.1038/s41598-026-46086-9}

\bibitem{deepa2026}
Deepa, B.G., Lokesh, C.K., Umamaheswari, D., et al.: An enhanced deep learning framework for DeepFake detection using EfficientNet-B3. Discover Computing \textbf{29}(1) (2026). \doi{10.1007/s10791-025-09890-x}

\bibitem{javiya2025}
Javiya, D., Jethava, G., Parmar, H., et al.: Enhancing deepfake detection with Adaptive-DCGAN and Lite-CNN. Discover Applied Sciences \textbf{7}(11) (2025). \doi{10.1007/s42452-025-07690-y}

\bibitem{soudy2024}
Soudy, A.H., Sayed, O., Tag-Elser, H., et al.: Deepfake detection using convolutional vision transformers and convolutional neural networks. Neural Computing and Applications \textbf{36}(31), 19759--19775 (2024). \doi{10.1007/s00521-024-10181-7}

\bibitem{magesh2025}
Magesh, A.P., Ramakrishnan, S.S.M., Arumuga Arun, R., et al.: Building an efficient Deep Fake detection system using CNNs and transformers. Discover Computing \textbf{28}(1) (2025). \doi{10.1007/s10791-025-09586-2}

\bibitem{castro2026}
Castro, F., Gattulli, V., Impedovo, D., et al.: DeepDect: an explainable AI platform for face swapping and face generation DeepFake detection. Multimedia Tools and Applications \textbf{85}(5) (2026). \doi{10.1007/s11042-026-21672-1}

\bibitem{mansoor2025}
Mansoor, N., Iliev, A.I.: Explainable AI for DeepFake detection. Applied Sciences \textbf{15}(2), 725 (2025). \doi{10.3390/app15020725}

\bibitem{momin2025}
Momin, M.S., Sufian, A., Barman, D., et al.: Explainable deepfake detection across different modalities: an overview of methods and challenges. Image and Vision Computing \textbf{163}, 105738 (2025). \doi{10.1016/j.imavis.2025.105738}

\bibitem{tsigos2024}
Tsigos, K., Apostolidis, E., Baxevanakis, S., et al.: Towards quantitative evaluation of explainable AI methods for deepfake detection. In: Proc. MAD~'24, pp. 37--45 (2024). \doi{10.1145/3643491.3660292}

\bibitem{nimah2025}
Ni'mah, I., Wijayanti, R., Santosa, A., et al.: A simple contrastive embedding framework for low-resource fake news detection. Neural Computing and Applications \textbf{37}(26), 21407--21433 (2025). \doi{10.1007/s00521-025-11467-0}

\bibitem{rama2025}
Rama Moorthy, H., Avinash, N.J., Krishnaraj Rao, N.S., et al.: Dual stream graph augmented transformer model integrating BERT and GNNs for context aware fake news detection. Scientific Reports \textbf{15}(1) (2025). \doi{10.1038/s41598-025-05586-w}

\bibitem{kumarc2025}
Kumar, C., Bansal, M., Khan, M.A., et al.: Graph-augmented transformer ensemble framework for robust and scalable fake news detection in social media ecosystems. Scientific Reports \textbf{16}(1) (2025). \doi{10.1038/s41598-025-31653-3}

\bibitem{alqadi2025}
Alqadi, B.S., Alsuhibany, S.A., Yousafzai, S.N., et al.: Transfer learning driven fake news detection and classification using large language models. Scientific Reports \textbf{15}(1) (2025). \doi{10.1038/s41598-025-10670-2}

\bibitem{chaleh2025}
Chalehchaleh, R., Farahbakhsh, R., Crespi, N.: Addressing data scarcity in multilingual fake news detection: an LLM-based dataset augmentation approach. Social Network Analysis and Mining \textbf{15}(1) (2025). \doi{10.1007/s13278-025-01505-z}

\bibitem{sneha2026}
Sneha, S.G., Sen, A., Malik, S., et al.: BiLSTM-LIME: integrating NLP and advanced machine learning models for fake news detection. Discover Artificial Intelligence \textbf{6}(1) (2026). \doi{10.1007/s44163-026-00852-w}

\bibitem{men2024}
Men, X., Mariano, V.Y.: Explainable fake news detection based on BERT and SHAP applied to COVID-19. International Journal of Modern Education and Computer Science \textbf{16}(1), 11--22 (2024). \doi{10.5815/ijmecs.2024.01.02}

\bibitem{bezzaoui2025}
Bezzaoui, I., Stein, C., Weinhardt, C., et al.: Explainable AI for online disinformation detection: insights from a design science research project. Electronic Markets \textbf{35}(1) (2025). \doi{10.1007/s12525-025-00799-3}

\bibitem{naude2023}
Naud\'{e}, M., Adebayo, K.J., Nanda, R.: A machine learning approach to detecting fraudulent job types. AI \& SOCIETY \textbf{38}(2), 1013--1024 (2023). \doi{10.1007/s00146-022-01469-0}

\bibitem{franceschini2026}
Franceschini, I., Wyndham, C., Le, L., et al.: Imperfect victims: comparing Chinese and Vietnamese survivors of forced criminality in Southeast Asia's online scam industry. Asian Journal of Criminology \textbf{21}(1) (2026). \doi{10.1007/s11417-026-09495-3}

\bibitem{asiama2026}
Asiama, A.A.: Pathways of intention: narratives of future withdrawal from cyber fraud among Ghanaian youth. Journal of Developmental and Life-Course Criminology \textbf{12}(1) (2026). \doi{10.1007/s40865-026-00306-y}

\bibitem{abbas2024}
Abbas, F., Taeihagh, A.: Unmasking deepfakes: a systematic review of deepfake detection and generation techniques using artificial intelligence. Expert Systems with Applications \textbf{252}, 124260 (2024). \doi{10.1016/j.eswa.2024.124260}

\bibitem{lv2025}
Lv, J., Gao, Y., Li, L., et al.: Multi-modal fake news detection: a comprehensive survey on deep learning technology, advances, and challenges. Journal of King Saud University Computer and Information Sciences \textbf{37}(9) (2025). \doi{10.1007/s44443-025-00317-7}

\bibitem{omar2024}
Omar, K., Sakr, R.H., Alrahmawy, M.F.: An ensemble of CNNs with self-attention mechanism for DeepFake video detection. Neural Computing and Applications \textbf{36}(6), 2749--2765 (2024). \doi{10.1007/s00521-023-09196-3}

\bibitem{duan2025}
Duan, H., Jiang, Q., Jin, X., et al.: Mf-net: multi-feature fusion network based on two-stream extraction and multi-scale enhancement for face forgery detection. Complex \& Intelligent Systems \textbf{11}(1) (2025). \doi{10.1007/s40747-024-01634-6}

\bibitem{gk2025}
G.K.M, Faizz Ahmad, K.S., Pamidimukkala, S.G., et al.: Hybrid optimization driven fake news detection using reinforced transformer models. Scientific Reports \textbf{15}(1) (2025). \doi{10.1038/s41598-025-99936-3}

\bibitem{almandouh2024}
Almandouh, M.E., Alrahmawy, M.F., Eisa, M., et al.: Ensemble based high performance deep learning models for fake news detection. Scientific Reports \textbf{14}(1) (2024). \doi{10.1038/s41598-024-76286-0}

\end{thebibliography}
\end{document}
"""

tex_path.write_text(body + old_bib, encoding="utf-8")
extra = Path("docs/sicacs2026_llncs/ALL_50_REFERENCES.txt")
if extra.exists():
    extra.unlink()

print("REVERTED OK")
print("bibitems", old_bib.count(r"\bibitem{"))
print("refXX left", len(re.findall(r"ref\d\d", body)))
print("complementary gone", "Complementary corpus" not in body)
