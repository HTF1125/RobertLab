import streamlit as st
from .base import BasePage

class AboutMe(BasePage):
    DESCRIPTION = """
        Results-driven quantitative strategy and financial model developer
        with a strong background in buy-side equity investing.
        Offering 6 years of experience in financial research and strategy
        development, with a focus on developing innovative investment solutions.
        Proficient in quantitative strategy development for institutional clients,
        with 2+ years of experience in delivering successful projects.
    """
    EMAIL = "hantianfeng@outlook.com"

    def load_page(self):

        st.write(self.DESCRIPTION)
        st.write("📫", self.EMAIL)

        # --- EXPERIENCE & QUALIFICATIONS ---
        st.write("\n")
        self.h3("Experience & Qualifications")
        self.divider()


        self.h4("Quantitative Strategy")
        st.markdown("- 📈 6 years of experience in developing quant strategies.")
        st.markdown("- 🖥️ Extensive expertise in Python and Excel for .")
        st.markdown("- 📊 Proficient in implementing financial and statistical principles in real-world applications.")

        self.h4("Team Collaboration")
        st.markdown("- 🤝 Excellent team player, fostering collaboration and communication.")
        st.markdown("- 🚀 Strong sense of initiative, taking ownership of tasks and driving them to completion.")
        st.markdown("- 🎯 Goal-oriented mindset, working towards achieving team objectives.")

        self.h4("Hard Skills")
        st.markdown("- 💻 Programming: Python, SQL, VBA")
        st.markdown("- 📚 Modeling: Portfolio Optimization, Factor & Risk Premia")
        st.markdown("- 🌍 Languages: Chinese, English, Korean, Japanese (Beginner)")

        self.h4("Additional Skills")
        st.markdown("- 🔍 Advanced knowledge of data analysis and visualization tools.")
        st.markdown("- 🏆 Proven track record of delivering high-quality results within deadlines.")
        st.markdown("- 📖 Continuously learning and keeping up with the latest trends and technologies.")


        # --- WORK HISTORY ---
        st.write("\n")
        self.h3("Work History")
        self.divider()

        # --- JOB 1 ---
        self.h4("🚧 Quantitative Strategist | Dneuro Inc.")
        st.markdown("05/2021 - 05/2023")
        st.write(
            """
            -► Spearheaded the development of a comprehensive Robo-Advisor project,
              encompassing all aspects of wealth management, including goal-based
              dynamic asset allocation, market regime analysis, macro factors, and
              asset selection methodologies. Integrated behavioral finance to
              enhance client experience.

            -► Led the US equity factor library construction project, developing
              Python calculations for over 100 fundamental factors. Implemented a
              database operations module using MariaDB to facilitate efficient
              data management.

            -► Provided consulting services for OCIO Strategic Asset Allocation,
              utilizing simulations and portfolio optimization techniques.
              Developed a user-friendly web portal for client interactions and
              customized asset allocation strategies.
            """
        )

        self.divider()

        # --- JOB 2 ---
        self.h4("🚧 Global Solutions Specialist | Woori Asset Management Corp.")
        st.markdown("03/2017 - 05/2021")
        st.write(
            """
            -► Contributed to quantitative research and played a key role in
              developing models for optimizing global multi-asset portfolios.
              Leveraged advanced quantitative techniques to analyze market data,
              identify trends, and enhance portfolio performance and risk management.

            -► Conducted global research, focusing on US and Chinese equities.
              Built multiple financial analysis models in Excel (VBA) using
              Bloomberg and Wind terminals.

            -► Managed liquidity positions for all global equity funds, ensuring
              efficient execution of Forex hedges using futures and forward
              contracts. Monitored market conditions and implemented hedging
              strategies to mitigate currency risks and optimize fund performance.
            """
        )

        st.warning("Feel free to reach out to discuss potential opportunities!")

